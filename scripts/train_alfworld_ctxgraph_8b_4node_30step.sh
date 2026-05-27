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
# ContextGraph on ALFWorld @real, 4B / 2 nodes / 4h.
# Pivot from HotpotQA (4B-Instruct saturated at 0.7) AND from ALFWorld
# @hard (cold-start at 0%, all rollouts get reward 0, no gradient).
# @real shows admissible commands so 4B baseline lands ~0.35-0.50 with
# real headroom for RL. Reads data/alfworld_{train,test}.parquet which
# must have ability=ALFWorld@real (regenerate via
# `python scripts/make_alfworld_data.py --n_train 300 --n_val 80`).
#
# Why 2-node 4h: matches the experimentation cadence used for the
# HotpotQA 4B variants in this iteration, fits a single idev session.
# Production headline run is still train_alfworld_ctxgraph_4b_4node_16h.sh
# (this is the fast-iteration sibling).
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

# ── WANDB (optional) ──
if [ -n "${WORK:-}" ] && [ -f "$WORK/.wandb_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.wandb_env"
fi
export WANDB_API_KEY=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

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
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=NVL

# ── Project paths ──
PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
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

if [ -n "${WANDB_API_KEY:-}" ]; then
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

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="ctxgraph_alfworld_${ALFWORLD_MODE}_8b_4n_p4096_r8192_30step_${TS}"

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  ContextGraph (isolated) on ALFWorld @${ALFWORLD_MODE} (8B, 4 nodes, 30 steps)"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Worker(s): ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  ALFWORLD_DATA:  $ALFWORLD_DATA"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: ALFWorld artefacts must already exist ──
probe "checking ALFWorld artefacts (mode=$ALFWORLD_MODE)"
TRAIN_PARQUET="$PROJECT_ROOT/data/alfworld_${ALFWORLD_MODE}_train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data/alfworld_${ALFWORLD_MODE}_test.parquet"
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
TRAINER_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
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
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  ray start --head --node-ip-address='"$NODE0_IP"' --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
' &
RAY_HEAD_PID=$!
sleep 20
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
    export FLASHINFER_WORKSPACE_BASE=/tmp
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    ray start --address='"${NODE0_IP}:6379"' --num-cpus=70 --num-gpus=1 --block
  ' &
  WORKER_PIDS+=("$!")
  sleep 5
done
sleep 20
probe "all Ray workers launched, cluster settling"

cleanup() {
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

export RAY_ADDRESS=${NODE0_IP}:6379
probe "querying ray status"
ray status || echo "WARN: ray status check failed"

echo "=============================================================="
echo "  Launching ContextGraph FoldGRPO training on ALFWorld"
echo "  vLLM gpu_memory_utilization=0.55 (no embedder co-located, all GPU mem available)"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min before first wandb log)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" --chdir="$PROJECT_ROOT" \
  --export=ALL,ALFWORLD_DATA="$ALFWORLD_DATA" \
  python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=4096 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
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
  data.train_files=data/alfworld_${ALFWORLD_MODE}_train.parquet \
  data.val_files=data/alfworld_${ALFWORLD_MODE}_test.parquet \
  data.train_batch_size=16 \
  data.max_prompt_length=4096 \
  data.max_response_length=8192 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=12288 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld_graph \
  +actor_rollout_ref.rollout.plugin.max_turn=20 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=512 \
  +actor_rollout_ref.rollout.plugin.max_session=3 \
  +actor_rollout_ref.rollout.plugin.val_max_session=3 \
  +actor_rollout_ref.rollout.plugin.session_timeout=300 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=2048 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope,graph]' \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=20 \
  +actor_rollout_ref.rollout.plugin.val_response_length=8192 \
  trainer.val_before_train=True \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=30 \
  trainer.test_freq=999 \
  trainer.save_freq=15 \
  trainer.default_local_dir=${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/$EXPERIMENT_NAME \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER"
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
