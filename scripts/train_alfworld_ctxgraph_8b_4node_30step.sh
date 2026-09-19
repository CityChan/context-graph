#!/bin/bash
#SBATCH -J cg-alf-8b-4n
#SBATCH -o logs/cg-alf-8b-4n.%j.out
#SBATCH -e logs/cg-alf-8b-4n.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# ContextGraph on ALFWorld @real, 8B / 4 nodes / 4h.
# Pivot from ALFWorld @hard (cold-start at 0%, all rollouts get reward 0,
# no gradient).
# @real shows admissible commands so the baseline has
# real headroom for RL. Reads data/alfworld_{train,test}.parquet which
# must have ability=ALFWorld@real (regenerate via
# `python scripts/make_alfworld_data.py --n_train 300 --n_val 80`).
#
# Why 4-node 4h: fits a short idev diagnostic session while still exercising
# the distributed training path.
#
# K=3 trick already removed from agents/graph_agent_isolated.py
# (commit f211fbd) — paper-faithful design.
#
# Pre-flight (one-time, before this run):
#   (login) pip install textworld alfworld
#   (login) alfworld-download                 # writes to ~/.cache/alfworld
#   (login) python scripts/make_alfworld_data.py --n_train 300 --n_val 80
#                                              ^^ NO --hard flag for this script
#                                              (admissible commands shown = @real).
#                                              Verify ability=ALFWorld@real in the
#                                              parquet before launching:
#                                                python -c "import pandas as pd; \
#                                                  print(pd.read_parquet( \
#                                                  'data/alfworld_train.parquet' \
#                                                  )['ability'].value_counts())"
#
# Pairs with train_alfworld_foldagent_4b_2node_4h.sh — same backbone,
# same data, only the agent loop + workflow + reward differ:
#   agent loop  : context_graph_isolated_agent
#   workflow    : alfworld_graph
#   reward      : flat + scope + graph (outcome-only graph reward)
#   extras      : lambda_compact 0.1, lambda_cost 0.005
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Vista cache redirects (avoid NFS flock) ──
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# HF datasets cache MUST live on local /tmp, not NFS — datasets writes
# .incomplete/ dirs then renames, which races on Vista's /work.
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_$$
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-180}
RAY_HEAD_SETTLE_SECONDS=${RAY_HEAD_SETTLE_SECONDS:-30}
RAY_WORKER_STAGGER_SECONDS=${RAY_WORKER_STAGGER_SECONDS:-8}
RAY_CLUSTER_SETTLE_SECONDS=${RAY_CLUSTER_SETTLE_SECONDS:-30}
RAY_STATUS_TIMEOUT_SECONDS=${RAY_STATUS_TIMEOUT_SECONDS:-300}
RAY_STATUS_POLL_SECONDS=${RAY_STATUS_POLL_SECONDS:-10}

# ── WANDB (optional) ──
if [ -n "${WORK:-}" ] && [ -f "$WORK/.wandb_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.wandb_env"
fi

# ── Conda + CUDA ──
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LIBRARY_PATH:-}
export CPATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/include:${CPATH:-}

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCHDYNAMO_DISABLE=1
# vLLM V1 applies structured-output masks through an xgrammar
# torch.compile-decorated kernel. The Vista build can emit invalid Inductor
# Python for it, so default this small operation to eager execution.
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=NVL

# ── Project paths ──
PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export ALFWORLD_MAX_ADMISSIBLE_DISPLAY=${ALFWORLD_MAX_ADMISSIBLE_DISPLAY:-0}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── Node info ──
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

if [ "$NUM_NODES" -ne 4 ]; then
  echo "Expected 4 nodes (set #SBATCH -N 4 or use idev -N 4), got $NUM_NODES"
  exit 1
fi

if [ "${ALFWORLD_DISABLE_WANDB:-0}" = "1" ]; then
  TRAINER_LOGGER='["console"]'
elif [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
else
  TRAINER_LOGGER='["console"]'
fi

# ── ALFWorld mode (real / hard) — switch without regenerating data ──
# make_alfworld_data.py writes both alfworld_real_* and alfworld_hard_*
# (and the _graph_ variants) in one go. Pick which one this run uses
# by setting ALFWORLD_MODE. Default: real (Goldilocks for 4B).
ALFWORLD_MODE=${ALFWORLD_MODE:-real}
if [ "$ALFWORLD_MODE" != "real" ] && [ "$ALFWORLD_MODE" != "hard" ]; then
  echo "ERROR: ALFWORLD_MODE must be 'real' or 'hard' (got '$ALFWORLD_MODE')"
  exit 1
fi
ALFWORLD_TRAIN_MODULE=${ALFWORLD_TRAIN_MODULE:-scripts.train_graph}
ALFWORLD_AGENT_LOOP=${ALFWORLD_AGENT_LOOP:-context_graph_isolated_agent}
ALFWORLD_WORKFLOW=${ALFWORLD_WORKFLOW:-alfworld_graph}
ALFWORLD_PROCESS_REWARD=${ALFWORLD_PROCESS_REWARD:-[flat,scope,graph]}
case "$ALFWORLD_AGENT_LOOP" in
  react_agent) DEFAULT_METHOD_LABEL=ReAct ;;
  fold_agent) DEFAULT_METHOD_LABEL=FoldAgent ;;
  context_graph_isolated_agent) DEFAULT_METHOD_LABEL="ContextGraph (isolated)" ;;
  *) DEFAULT_METHOD_LABEL=$ALFWORLD_AGENT_LOOP ;;
esac
ALFWORLD_METHOD_LABEL=${ALFWORLD_METHOD_LABEL:-$DEFAULT_METHOD_LABEL}
case "$ALFWORLD_WORKFLOW" in
  alfworld) DEFAULT_DATA_VARIANT=alfworld ;;
  alfworld_branch) DEFAULT_DATA_VARIANT=alfworld_branch ;;
  alfworld_graph) DEFAULT_DATA_VARIANT=alfworld_graph ;;
  *)
    echo "ERROR: unsupported ALFWORLD_WORKFLOW=$ALFWORLD_WORKFLOW"
    exit 1
    ;;
esac
ALFWORLD_DATA_VARIANT=${ALFWORLD_DATA_VARIANT:-$DEFAULT_DATA_VARIANT}

ALFWORLD_VAL_ONLY=${ALFWORLD_VAL_ONLY:-False}
ALFWORLD_VAL_BEFORE_TRAIN=${ALFWORLD_VAL_BEFORE_TRAIN:-True}
ALFWORLD_TOTAL_STEPS=${ALFWORLD_TOTAL_STEPS:-30}
ALFWORLD_TRAIN_BATCH_SIZE=${ALFWORLD_TRAIN_BATCH_SIZE:-16}
ALFWORLD_PPO_MINI_BATCH_SIZE=${ALFWORLD_PPO_MINI_BATCH_SIZE:-$ALFWORLD_TRAIN_BATCH_SIZE}
ALFWORLD_ROLLOUT_N=${ALFWORLD_ROLLOUT_N:-4}
ALFWORLD_PROMPT_LENGTH=${ALFWORLD_PROMPT_LENGTH:-4096}
ALFWORLD_RESPONSE_LENGTH=${ALFWORLD_RESPONSE_LENGTH:-8192}
ALFWORLD_MAX_TOKEN_LEN_PER_GPU=${ALFWORLD_MAX_TOKEN_LEN_PER_GPU:-12288}
ALFWORLD_MAX_TURN=${ALFWORLD_MAX_TURN:-20}
ALFWORLD_VAL_MAX_TURN=${ALFWORLD_VAL_MAX_TURN:-$ALFWORLD_MAX_TURN}
ALFWORLD_TURN_MAX_NEW_TOKENS=${ALFWORLD_TURN_MAX_NEW_TOKENS:-512}
ALFWORLD_BRANCH_LEN=${ALFWORLD_BRANCH_LEN:-2048}
ALFWORLD_STRUCTURED_GRAPH_CONTROLLER=${ALFWORLD_STRUCTURED_GRAPH_CONTROLLER:-False}
ALFWORLD_CONTROLLER_OWNED_TOOL_FORMATTING=${ALFWORLD_CONTROLLER_OWNED_TOOL_FORMATTING:-$ALFWORLD_STRUCTURED_GRAPH_CONTROLLER}
ALFWORLD_CONTROLLER_ACTION_POLICY=${ALFWORLD_CONTROLLER_ACTION_POLICY:-balanced}
ALFWORLD_CONTROLLER_ALLOW_PASS=${ALFWORLD_CONTROLLER_ALLOW_PASS:-False}
ALFWORLD_CONSOLIDATION_INTERVAL=${ALFWORLD_CONSOLIDATION_INTERVAL:-0}
ALFWORLD_ENABLE_RETRIEVAL_MEMORY=${ALFWORLD_ENABLE_RETRIEVAL_MEMORY:-True}
ALFWORLD_INJECT_GRAPH_STATE_AFTER_ACTION=${ALFWORLD_INJECT_GRAPH_STATE_AFTER_ACTION:-True}
ALFWORLD_MAX_SESSION=${ALFWORLD_MAX_SESSION:-3}
ALFWORLD_VAL_MAX_SESSION=${ALFWORLD_VAL_MAX_SESSION:-$ALFWORLD_MAX_SESSION}
ALFWORLD_SESSION_TIMEOUT=${ALFWORLD_SESSION_TIMEOUT:-300}
ALFWORLD_TRAINER_RESUME_MODE=${ALFWORLD_TRAINER_RESUME_MODE:-auto}
ALFWORLD_TRAIN_MAX_SAMPLES=${ALFWORLD_TRAIN_MAX_SAMPLES:-}
ALFWORLD_VAL_MAX_SAMPLES=${ALFWORLD_VAL_MAX_SAMPLES:-}
ALFWORLD_TEST_FREQ=${ALFWORLD_TEST_FREQ:-999}
ALFWORLD_SAVE_FREQ=${ALFWORLD_SAVE_FREQ:-15}

case "$ALFWORLD_CONTROLLER_ACTION_POLICY" in
  balanced|structural) ;;
  *)
    echo "ERROR: ALFWORLD_CONTROLLER_ACTION_POLICY must be balanced or structural; got $ALFWORLD_CONTROLLER_ACTION_POLICY"
    exit 1
    ;;
esac
case "$ALFWORLD_TRAINER_RESUME_MODE" in
  auto|disable) ;;
  *)
    echo "ERROR: ALFWORLD_TRAINER_RESUME_MODE must be auto or disable; got $ALFWORLD_TRAINER_RESUME_MODE"
    exit 1
    ;;
esac

EXTRA_DATA_ARGS=()
if [ -n "$ALFWORLD_TRAIN_MAX_SAMPLES" ]; then
  EXTRA_DATA_ARGS+=(data.train_max_samples="$ALFWORLD_TRAIN_MAX_SAMPLES")
fi
if [ -n "$ALFWORLD_VAL_MAX_SAMPLES" ]; then
  EXTRA_DATA_ARGS+=(data.val_max_samples="$ALFWORLD_VAL_MAX_SAMPLES")
fi

TS=$(date +%Y%m%d_%H%M%S)
RUN_SUFFIX="step${ALFWORLD_TOTAL_STEPS}"
RUN_KIND="FoldGRPO training"
if [ "$ALFWORLD_VAL_ONLY" = "True" ] || [ "$ALFWORLD_VAL_ONLY" = "true" ]; then
  RUN_SUFFIX="valonly"
  RUN_KIND="validation-only evaluation"
fi
EXPERIMENT_NAME=${ALFWORLD_EXPERIMENT_NAME:-"${ALFWORLD_AGENT_LOOP}_${ALFWORLD_WORKFLOW}_${ALFWORLD_MODE}_8b_4n_p${ALFWORLD_PROMPT_LENGTH}_r${ALFWORLD_RESPONSE_LENGTH}_${RUN_SUFFIX}_${TS}"}

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

wait_for_ray_cluster() {
  local expected_nodes=$1
  local deadline=$((SECONDS + RAY_STATUS_TIMEOUT_SECONDS))
  local status_out active_nodes

  while [ "$SECONDS" -lt "$deadline" ]; do
    status_out=$(ray status 2>&1 || true)
    active_nodes=$(printf '%s\n' "$status_out" | awk '
      /^Active:/ {active=1; next}
      /^Pending:/ {active=0}
      active && /node_/ {count++}
      END {print count + 0}
    ')
    if [ "$active_nodes" -ge "$expected_nodes" ]; then
      printf '%s\n' "$status_out"
      return 0
    fi
    printf '+++ [%s] waiting for Ray nodes: active=%s expected=%s\n' \
      "$(date +%H:%M:%S)" "$active_nodes" "$expected_nodes"
    sleep "$RAY_STATUS_POLL_SECONDS"
  done

  echo "ERROR: Ray cluster did not reach $expected_nodes active nodes within ${RAY_STATUS_TIMEOUT_SECONDS}s."
  echo "Last ray status output:"
  printf '%s\n' "$status_out"
  return 1
}

echo "=============================================================="
echo "  ${ALFWORLD_METHOD_LABEL} on ALFWorld @${ALFWORLD_MODE} (8B, 4 nodes, steps=${ALFWORLD_TOTAL_STEPS})"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Worker(s): ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  ALFWORLD_DATA:  $ALFWORLD_DATA"
echo "  val_only:       $ALFWORLD_VAL_ONLY"
echo "  steps:          $ALFWORLD_TOTAL_STEPS"
echo "  samples:        train=${ALFWORLD_TRAIN_MAX_SAMPLES:-all} val=${ALFWORLD_VAL_MAX_SAMPLES:-all}"
echo "  tokens/turns:   prompt=$ALFWORLD_PROMPT_LENGTH response=$ALFWORLD_RESPONSE_LENGTH max_turn=$ALFWORLD_MAX_TURN val_max_turn=$ALFWORLD_VAL_MAX_TURN"
echo "  admissible:     official demangled commands, max_display=$ALFWORLD_MAX_ADMISSIBLE_DISPLAY"
echo "  agent/workflow: module=$ALFWORLD_TRAIN_MODULE loop=$ALFWORLD_AGENT_LOOP workflow=$ALFWORLD_WORKFLOW process_reward=$ALFWORLD_PROCESS_REWARD"
echo "  data variant:   $ALFWORLD_DATA_VARIANT"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: ALFWorld artefacts must already exist ──
probe "checking ALFWorld artefacts (mode=$ALFWORLD_MODE)"
TRAIN_PARQUET="$PROJECT_ROOT/data/${ALFWORLD_DATA_VARIANT}_${ALFWORLD_MODE}_train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data/${ALFWORLD_DATA_VARIANT}_${ALFWORLD_MODE}_test.parquet"
JSON_DIR="$ALFWORLD_DATA/json_2.1.1"
for f in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    echo
    echo "Stage prep, in order (login node):"
    echo "  pip install textworld alfworld"
    echo "  alfworld-download                 # writes to ~/.cache/alfworld"
    echo "  python scripts/make_alfworld_data.py --n_train 300 --n_val 80"
    echo "  # (default --mode=both writes alfworld_{real,hard}_* in one shot)"
    exit 1
  fi
done
if [ ! -d "$JSON_DIR" ]; then
  echo "ERROR: ALFWorld game files not found under $JSON_DIR"
  echo "Run: alfworld-download"
  exit 1
fi
probe "ALFWorld artefacts ok"

# ── Pre-flight: trainer weights must be present (offline, no auto-dl) ──
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
probe "checking HF model cache for $MODEL_PATH"
if [ -d "$MODEL_PATH" ]; then
  TRAINER_CACHE_DIR="$MODEL_PATH"
  if [ ! -s "$TRAINER_CACHE_DIR/config.json" ]; then
    echo "ERROR: local MODEL_PATH is missing config.json: $TRAINER_CACHE_DIR"
    exit 1
  fi
  if ! find -L "$TRAINER_CACHE_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit | grep -q .; then
    echo "ERROR: local MODEL_PATH has no non-empty Hugging Face weight files: $TRAINER_CACHE_DIR"
    exit 1
  fi
else
  TRAINER_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
fi
if [ ! -d "$TRAINER_CACHE_DIR" ]; then
  echo "ERROR: $MODEL_PATH not found at $TRAINER_CACHE_DIR"
  echo "       From a login node, run: hf download $MODEL_PATH"
  exit 1
fi
probe "trainer cache: $TRAINER_CACHE_DIR"

# ── Stale Ray cleanup on all allocated nodes ──
probe "ray stop sweep across $NUM_NODES nodes"
for node in "${NODELIST[@]}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -c '
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    ray stop -f >/dev/null 2>&1 || true
  ' || true
done
sleep 5
probe "ray stop sweep done"

# ── Sanity ──
probe "python sanity imports (cold torch import can take ~60s)"
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
python -c "import textworld; import alfworld; print('textworld + alfworld OK')"
probe "sanity imports done"

# ── Ray head ──
probe "starting Ray head on $NODE0"
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" bash -c '
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  hash -r
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
  export HF_HOME='"$HF_HOME"'
  export ALFWORLD_DATA='"$ALFWORLD_DATA"'
  export ALFWORLD_MAX_ADMISSIBLE_DISPLAY='"$ALFWORLD_MAX_ADMISSIBLE_DISPLAY"'
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export RAY_raylet_start_wait_time_s='"$RAY_raylet_start_wait_time_s"'
  ray start --head --node-ip-address='"$NODE0_IP"' --port=6379 \
    --num-cpus=70 --num-gpus=1 --include-dashboard=false --disable-usage-stats --block
' &
RAY_HEAD_PID=$!
sleep "$RAY_HEAD_SETTLE_SECONDS"
if ! kill -0 "$RAY_HEAD_PID" 2>/dev/null; then
  echo "ERROR: Ray head process exited before workers could join."
  wait "$RAY_HEAD_PID" || true
  exit 1
fi
probe "Ray head sleep done; launching $((NUM_NODES - 1)) workers"

# ── Ray workers ──
WORKER_PIDS=()
for i in $(seq 1 $((NUM_NODES - 1))); do
  WORKER_NODE=${NODELIST[$i]}
  srun --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c '
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    hash -r
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
    export HF_HOME='"$HF_HOME"'
    export ALFWORLD_DATA='"$ALFWORLD_DATA"'
    export ALFWORLD_MAX_ADMISSIBLE_DISPLAY='"$ALFWORLD_MAX_ADMISSIBLE_DISPLAY"'
    export FLASHINFER_WORKSPACE_BASE=/tmp
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export RAY_raylet_start_wait_time_s='"$RAY_raylet_start_wait_time_s"'
    ray start --address='"${NODE0_IP}:6379"' --num-cpus=70 --num-gpus=1 --block
  ' &
  WORKER_PIDS+=("$!")
  sleep "$RAY_WORKER_STAGGER_SECONDS"
done
sleep "$RAY_CLUSTER_SETTLE_SECONDS"
probe "all Ray workers launched, cluster settling"

cleanup() {
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

export RAY_ADDRESS=${NODE0_IP}:6379
probe "waiting for Ray cluster readiness"
if ! wait_for_ray_cluster "$NUM_NODES"; then
  echo "ERROR: Ray cluster did not start cleanly; aborting before trainer launch."
  exit 1
fi

echo "=============================================================="
echo "  Launching ${ALFWORLD_METHOD_LABEL} ${RUN_KIND} on ALFWorld"
echo "  controller: structured=${ALFWORLD_STRUCTURED_GRAPH_CONTROLLER} policy=${ALFWORLD_CONTROLLER_ACTION_POLICY} interval=${ALFWORLD_CONSOLIDATION_INTERVAL}"
echo "  memory: retrieval=${ALFWORLD_ENABLE_RETRIEVAL_MEMORY} graph_after_action=${ALFWORLD_INJECT_GRAPH_STATE_AFTER_ACTION} controller_pass=${ALFWORLD_CONTROLLER_ALLOW_PASS}"
echo "  resume_mode: ${ALFWORLD_TRAINER_RESUME_MODE}"
echo "  vLLM gpu_memory_utilization=0.55 (no embedder co-located, all GPU mem available)"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min before first wandb log)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" --chdir="$PROJECT_ROOT" \
  --export=ALL,ALFWORLD_DATA="$ALFWORLD_DATA",ALFWORLD_MAX_ADMISSIBLE_DISPLAY="$ALFWORLD_MAX_ADMISSIBLE_DISPLAY" \
  python -m "$ALFWORLD_TRAIN_MODULE" \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=${ALFWORLD_AGENT_LOOP} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=${ALFWORLD_PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${ALFWORLD_RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ALFWORLD_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=${ALFWORLD_ROLLOUT_N} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  data.train_files=data/${ALFWORLD_DATA_VARIANT}_${ALFWORLD_MODE}_train.parquet \
  data.val_files=data/${ALFWORLD_DATA_VARIANT}_${ALFWORLD_MODE}_test.parquet \
  data.train_batch_size=${ALFWORLD_TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${ALFWORLD_PROMPT_LENGTH} \
  data.max_response_length=${ALFWORLD_RESPONSE_LENGTH} \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${ALFWORLD_PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ALFWORLD_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${ALFWORLD_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=${ALFWORLD_WORKFLOW} \
  +actor_rollout_ref.rollout.plugin.structured_graph_controller=${ALFWORLD_STRUCTURED_GRAPH_CONTROLLER} \
  +actor_rollout_ref.rollout.plugin.controller_owned_tool_formatting=${ALFWORLD_CONTROLLER_OWNED_TOOL_FORMATTING} \
  +actor_rollout_ref.rollout.plugin.controller_action_policy=${ALFWORLD_CONTROLLER_ACTION_POLICY} \
  +actor_rollout_ref.rollout.plugin.controller_allow_pass=${ALFWORLD_CONTROLLER_ALLOW_PASS} \
  +actor_rollout_ref.rollout.plugin.enable_retrieval_memory=${ALFWORLD_ENABLE_RETRIEVAL_MEMORY} \
  +actor_rollout_ref.rollout.plugin.inject_graph_state_after_action=${ALFWORLD_INJECT_GRAPH_STATE_AFTER_ACTION} \
  +actor_rollout_ref.rollout.plugin.max_turn=${ALFWORLD_MAX_TURN} \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=${ALFWORLD_TURN_MAX_NEW_TOKENS} \
  +actor_rollout_ref.rollout.plugin.max_session=${ALFWORLD_MAX_SESSION} \
  +actor_rollout_ref.rollout.plugin.val_max_session=${ALFWORLD_VAL_MAX_SESSION} \
  +actor_rollout_ref.rollout.plugin.session_timeout=${ALFWORLD_SESSION_TIMEOUT} \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=${ALFWORLD_BRANCH_LEN} \
  +actor_rollout_ref.rollout.plugin.process_reward="$ALFWORLD_PROCESS_REWARD" \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
  +actor_rollout_ref.rollout.plugin.consolidation_interval=${ALFWORLD_CONSOLIDATION_INTERVAL} \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=${ALFWORLD_VAL_MAX_TURN} \
  +actor_rollout_ref.rollout.plugin.val_response_length=${ALFWORLD_RESPONSE_LENGTH} \
  trainer.val_before_train=${ALFWORLD_VAL_BEFORE_TRAIN} \
  trainer.val_only=${ALFWORLD_VAL_ONLY} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=${ALFWORLD_TOTAL_STEPS} \
  trainer.resume_mode=${ALFWORLD_TRAINER_RESUME_MODE} \
  trainer.test_freq=${ALFWORLD_TEST_FREQ} \
  trainer.save_freq=${ALFWORLD_SAVE_FREQ} \
  trainer.default_local_dir=${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/$EXPERIMENT_NAME \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER" \
  "${EXTRA_DATA_ARGS[@]}"
RC=$?
set -e

echo "=============================================================="
if [ $RC -eq 0 ]; then
  echo "  TRAIN RUN COMPLETED (exit 0)"
else
  echo "  TRAIN RUN FAILED OR CUT OFF (exit $RC)"
  echo "  Check ckpts under $PROJECT_ROOT/checkpoints/context-graph/$EXPERIMENT_NAME/"
fi
echo "  Finished: $(date)"
echo "=============================================================="

exit $RC
