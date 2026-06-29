#!/bin/bash
#SBATCH -J eval-sab-react-30b-inst-smoke
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/eval-sab-react-30b-inst-smoke.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/eval-sab-react-30b-inst-smoke.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 2:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 2-hour 8-NODE SMOKE eval for ScienceAgentBench vanilla ReAct
# @ Qwen3-30B-A3B-Instruct-2507. Paired with eval_sab_{fold,ctxgraph}_8b_8node_2h.sh
# for a quick 30B Instruct sanity check before attempting a full 102-task run.
#
# Topology (8 nodes, no search server):
#   NODELIST[0]   = Ray head + trainer rank 0
#   NODELIST[1-7] = Ray workers (8 FSDP/vLLM GPUs total = 1 per node)
#   No dedicated search node �?SAB uses an in-process Python sandbox
#   (envs/scienceagent_sandbox.py), not an external service.
#
# Expected wall clock:
#   model init        ~3-5 min
#   greedy val rollout defaults to SAB_VAL_MAX_SAMPLES=16 for smoke testing
#   override SAB_VAL_MAX_SAMPLES for smaller/larger subsets
#   no LLM judge (file-existence scorer in env.get_reward; Phase D2)
#   total: padded to 2h to leave room for 30B model/vLLM initialization.
#
# Pre-flight (one-time, login node):
#   # Download HF CSV
#   mkdir -p data
#   wget https://huggingface.co/datasets/osunlp/ScienceAgentBench/resolve/main/ScienceAgentBench.csv \
#        -O data/ScienceAgentBench.csv
#   # Unpack the full benchmark zip (password-protected �?request from upstream)
#   #   benchmark zip -> data/sab_benchmark/{datasets,eval_programs,gold_programs}/
#   unzip benchmark.zip -d data/sab_benchmark
#   # Build per-workflow parquets
#   python scripts/make_sab_data.py \
#       --csv data/ScienceAgentBench.csv \
#       --benchmark-dir data/sab_benchmark \
#       --out-dir data
#   # ensure the model is cached:
#   hf download Qwen/Qwen3-30B-A3B-Instruct-2507
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Self-owned log: line-buffered, written directly to Lustre so early output
#    survives even if slurmstepd's stdout buffer is lost on a hard kill / node fail.
#    Also records the submit-dir pwd to catch wrong-WorkDir relative -o failures.
LOG_ROOT=/work/09281/chc_1996/vista/context-graph/logs
SUBMIT_LOG_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}/logs"
EXTRA_LOG_ROOT="${EXTRA_LOG_ROOT:-}"
mkdir -p "$LOG_ROOT" "$SUBMIT_LOG_ROOT"
if [ -n "$EXTRA_LOG_ROOT" ]; then mkdir -p "$EXTRA_LOG_ROOT"; fi
SELF_LOG_NAME="${SLURM_JOB_NAME:-sab}.${SLURM_JOB_ID:-local}.self.log"
TEE_TARGETS=("$LOG_ROOT/$SELF_LOG_NAME")
if [ "$SUBMIT_LOG_ROOT" != "$LOG_ROOT" ]; then TEE_TARGETS+=("$SUBMIT_LOG_ROOT/$SELF_LOG_NAME"); fi
if [ -n "$EXTRA_LOG_ROOT" ] && [ "$EXTRA_LOG_ROOT" != "$LOG_ROOT" ] && [ "$EXTRA_LOG_ROOT" != "$SUBMIT_LOG_ROOT" ]; then TEE_TARGETS+=("$EXTRA_LOG_ROOT/$SELF_LOG_NAME"); fi
exec > >(stdbuf -oL tee -a "${TEE_TARGETS[@]}") 2>&1
echo "+++ [self-log] host=$(hostname -s) date=$(date) job=${SLURM_JOB_ID:-NA} submit_pwd=$(pwd) log_targets=${TEE_TARGETS[*]}"
export SELF_LOG_PATH="${TEE_TARGETS[0]}"

# ── Vista cache redirects (avoid NFS flock) ──
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

# Real per-task eval (eval_programs/<script> -> [0,1] success) vs the
# Phase-D2 file-existence placeholder. Default 0 (placeholder). Set
# SAB_REAL_EVAL=1 at submit time for paper-grade scoring. Exported onto
# every Ray node below because get_reward reads it inside AgentLoopWorker.
export SAB_REAL_EVAL=${SAB_REAL_EVAL:-0}
export SAB_EXPOSE_EVAL_CONTRACT=${SAB_EXPOSE_EVAL_CONTRACT:-0}
export SAB_INTERACTIVE_EVAL_FEEDBACK=${SAB_INTERACTIVE_EVAL_FEEDBACK:-0}
# Qwen3 Thinking models default to long <think> traces. For tool-use eval,
# disable thinking in the chat template unless explicitly overridden.
export QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-False}

# WANDB: source credentials from explicit env first, then common Vista/project locations.
WANDB_ENV_SOURCE="env"
if [ -z "${WANDB_API_KEY:-}" ]; then
  for wandb_env in "${WORK:-}/.wandb_env" /work/09281/chc_1996/vista/.wandb_env /work/09281/chc_1996/vista/context-graph/.wandb_env "$HOME/.wandb_env"; do
    if [ -n "$wandb_env" ] && [ -f "$wandb_env" ]; then
      # shellcheck disable=SC1090
      source "$wandb_env"
      WANDB_ENV_SOURCE="$wandb_env"
      break
    fi
  done
fi
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_DIR=${WANDB_DIR:-/work/09281/chc_1996/vista/context-graph/wandb}
mkdir -p "$WANDB_DIR"

# ── Conda + CUDA ──
set +u  # conda activation scripts reference unbound vars (PS1, _CE_CONDA) -> set -u would kill us silently
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
set -u
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
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Per-trajectory sandbox workdir root (scratch is fastest on Vista)
export SAB_WORKDIR_ROOT=${SAB_WORKDIR_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/sab_workdirs}
mkdir -p "$SAB_WORKDIR_ROOT"

# ── Node info ──
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

if [ "$NUM_NODES" -ne 8 ]; then
  echo "Expected 8 nodes (set #SBATCH -N 8 or use idev -N 8), got $NUM_NODES"
  exit 1
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$NUM_NODES}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:-16}
if (( TRAIN_BATCH_SIZE % NUM_NODES != 0 )); then
  echo "ERROR: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by NUM_NODES=$NUM_NODES"
  exit 1
fi
if (( PPO_MINI_BATCH_SIZE % NUM_NODES != 0 )); then
  echo "ERROR: PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE must be divisible by NUM_NODES=$NUM_NODES"
  exit 1
fi

if [ "${SAB_DISABLE_WANDB:-0}" = "1" ]; then
  TRAINER_LOGGER='["console"]'
  probe_msg="wandb disabled by SAB_DISABLE_WANDB=1"
elif [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  probe_msg="wandb enabled (key length=${#WANDB_API_KEY}, source=$WANDB_ENV_SOURCE, dir=$WANDB_DIR)"
elif [ -f "$HOME/.netrc" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  probe_msg="wandb enabled via $HOME/.netrc (dir=$WANDB_DIR)"
else
  TRAINER_LOGGER='["console"]'
  probe_msg="WARNING: no WANDB_API_KEY or $HOME/.netrc found; eval will only log to console"
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="eval_react_sab_30b_instruct_8n_smoke_${TS}"

export EXPERIMENT_NAME
export WANDB_RUN_ID=${WANDB_RUN_ID:-$EXPERIMENT_NAME}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}

wandb_status() {
  local phase="$1"
  local rc="${2:-0}"
  if [[ "$TRAINER_LOGGER" != *wandb* ]]; then
    return 0
  fi
  WANDB_PHASE="$phase" WANDB_STATUS_RC="$rc" python - <<'PY' || true
import os
import time

try:
    import wandb

    run = wandb.init(
        project="context-graph",
        name=os.environ.get("WANDB_NAME") or os.environ.get("EXPERIMENT_NAME"),
        id=os.environ.get("WANDB_RUN_ID"),
        resume=os.environ.get("WANDB_RESUME", "allow"),
        dir=os.environ.get("WANDB_DIR"),
        reinit=True,
    )
    phase = os.environ.get("WANDB_PHASE", "unknown")
    rc = int(os.environ.get("WANDB_STATUS_RC", "0"))
    wandb.log({
        "sab_script/heartbeat": 1,
        "sab_script/rc": rc,
        "sab_script/time": time.time(),
    }, step=0)
    run.summary["sab_script/phase"] = phase
    run.summary["sab_script/rc"] = rc
    run.summary["sab_script/local_self_log"] = os.environ.get("SELF_LOG_PATH", "")
    log_path = os.environ.get("SELF_LOG_PATH", "")
    if log_path and os.path.exists(log_path):
        wandb.save(log_path, policy="now")
    run.finish()
except Exception as exc:
    print(f"WARN: wandb status log failed: {exc}")
PY
}

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  SMOKE EVAL: ReAct (code) on ScienceAgentBench (Qwen3-30B-A3B-Instruct-2507, 8 nodes, 32K-resp, val_only=True)"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Workers: ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  Experiment:     $EXPERIMENT_NAME"
echo "  Sandbox workdir root: $SAB_WORKDIR_ROOT"
echo "  Logger: ${probe_msg}"
echo "  Batch sizes:    train=$TRAIN_BATCH_SIZE ppo_mini=$PPO_MINI_BATCH_SIZE"
echo "  Smoke samples:  $SAB_VAL_MAX_SAMPLES"
echo "  Qwen thinking:  $QWEN_ENABLE_THINKING"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: data parquets must exist ──
probe "checking ScienceAgentBench artefacts"
VAL_PARQUET="$PROJECT_ROOT/data/sab_test_code.parquet"
if [ ! -f "$VAL_PARQUET" ]; then
  echo "ERROR: missing $VAL_PARQUET"
  echo "       Run: python scripts/make_sab_data.py --csv data/ScienceAgentBench.csv \\"
  echo "                 --benchmark-dir data/sab_benchmark --out-dir data"
  exit 1
fi
# verl wants a train_files path too even with val_only=True; reuse the same file.
TRAIN_PARQUET="$VAL_PARQUET"
probe "SAB parquet: $VAL_PARQUET"

# ── Pre-flight: model weights must be cached (offline) ──
probe "checking model cache"
TRAINER_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
if [ ! -d "$TRAINER_CACHE_DIR" ]; then
  echo "ERROR: $MODEL_PATH not cached at $TRAINER_CACHE_DIR"
  echo "       Login node: hf download $MODEL_PATH"
  exit 1
fi
probe "trainer cache: $TRAINER_CACHE_DIR"

# ── Topology: NODE0 = Ray head + trainer rank 0; NODE1-4 = Ray workers ──
TRAINER_HEAD_NODE=${NODELIST[0]}
TRAINER_HEAD_IP=$(getent hosts "$TRAINER_HEAD_NODE" | awk '{print $1}')
echo "  Trainer Ray head:      $TRAINER_HEAD_NODE ($TRAINER_HEAD_IP)"
echo "  Trainer workers:       ${NODELIST[@]:1}"

# ── Stale Ray cleanup on all nodes ──
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

# ── Sanity imports ──
probe "python sanity imports"
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
python -c "from envs.scienceagent_sandbox import CodeSandbox; from envs.scienceagent_env import ScienceAgentEnv; print('SAB env OK')"
probe "sanity imports done"

# ── Ray head on NODELIST[0] ──
probe "starting Ray head on $TRAINER_HEAD_NODE"
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" bash -c '
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  hash -r
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
  export HF_HOME='"$HF_HOME"'
  export HF_HUB_CACHE='"$HF_HUB_CACHE"'
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export SAB_WORKDIR_ROOT='"$SAB_WORKDIR_ROOT"'
  export SAB_REAL_EVAL='"$SAB_REAL_EVAL"'
  export SAB_EXPOSE_EVAL_CONTRACT='"$SAB_EXPOSE_EVAL_CONTRACT"'
  export SAB_INTERACTIVE_EVAL_FEEDBACK='"$SAB_INTERACTIVE_EVAL_FEEDBACK"'
  export QWEN_ENABLE_THINKING='"$QWEN_ENABLE_THINKING"'
  ray start --head --node-ip-address='"$TRAINER_HEAD_IP"' --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
' &
RAY_HEAD_PID=$!
sleep 20
probe "Ray head sleep done; launching $((NUM_NODES - 1)) trainer workers"

# ── Ray workers on NODELIST[1..N-1] ──
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
    export HF_HUB_CACHE='"$HF_HUB_CACHE"'
    export FLASHINFER_WORKSPACE_BASE=/tmp
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export SAB_WORKDIR_ROOT='"$SAB_WORKDIR_ROOT"'
    export SAB_REAL_EVAL='"$SAB_REAL_EVAL"'
    export SAB_EXPOSE_EVAL_CONTRACT='"$SAB_EXPOSE_EVAL_CONTRACT"'
    export SAB_INTERACTIVE_EVAL_FEEDBACK='"$SAB_INTERACTIVE_EVAL_FEEDBACK"'
  export QWEN_ENABLE_THINKING='"$QWEN_ENABLE_THINKING"'
    ray start --address='"${TRAINER_HEAD_IP}:6379"' --num-cpus=70 --num-gpus=1 --block
  ' &
  WORKER_PIDS+=("$!")
  sleep 5
done
sleep 20
probe "all $((NUM_NODES - 1)) trainer workers launched, cluster settling"

cleanup() {
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

export RAY_ADDRESS=${TRAINER_HEAD_IP}:6379
probe "querying ray status"
ray status || echo "WARN: ray status check failed"

echo "=============================================================="
echo "  Launching ReAct (code) SMOKE eval (8 nodes, 32K resp, ScienceAgentBench smoke subset)"
echo "  default_agent_loop=react_agent_code  workflow=code  process_reward=[flat]"
echo "  vLLM gpu_memory_utilization=0.55 + FP8 rollout + FSDP CPU offload (30B)"
echo "  val_only=True (one val pass on ${SAB_VAL_MAX_SAMPLES} samples from sab_test_code.parquet then exit; no training)"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~10-15 min)"
wandb_status trainer_launch 0

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" --chdir="$PROJECT_ROOT" \
  --export=ALL,SAB_WORKDIR_ROOT="$SAB_WORKDIR_ROOT",SAB_REAL_EVAL="$SAB_REAL_EVAL",SAB_EXPOSE_EVAL_CONTRACT="$SAB_EXPOSE_EVAL_CONTRACT",SAB_INTERACTIVE_EVAL_FEEDBACK="$SAB_INTERACTIVE_EVAL_FEEDBACK",QWEN_ENABLE_THINKING="$QWEN_ENABLE_THINKING" \
  python -m scripts.train_sab \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=react_agent_code \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  +actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=16384 \
  actor_rollout_ref.rollout.response_length=24576 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
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
  actor_rollout_ref.actor.grad_clip=0.5 \
  actor_rollout_ref.actor.kl_loss_coef=0.0005 \
  data.train_files=data/sab_test_code.parquet \
  data.val_files=data/sab_test_code.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.train_max_samples=$SAB_VAL_MAX_SAMPLES \
  data.val_max_samples=$SAB_VAL_MAX_SAMPLES \
  data.max_prompt_length=16384 \
  data.max_response_length=24576 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=40960 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=code \
  +actor_rollout_ref.rollout.plugin.max_turn=32 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.sandbox_timeout=60 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat]' \
  +actor_rollout_ref.rollout.plugin.val_max_turn=32 \
  +actor_rollout_ref.rollout.plugin.val_response_length=24576 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=$NUM_NODES \
  trainer.total_training_steps=1 \
  trainer.test_freq=999 \
  trainer.save_freq=999 \
  trainer.default_local_dir=${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/$EXPERIMENT_NAME \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER"
RC=$?
set -e
wandb_status trainer_exit "$RC"

echo "=============================================================="
if [ $RC -eq 0 ]; then
  echo "  EVAL RUN COMPLETED (exit 0)"
else
  echo "  EVAL RUN FAILED (exit $RC) �?check above for first error"
fi
echo "  Finished: $(date)"
echo "=============================================================="

exit $RC
