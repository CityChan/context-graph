#!/bin/bash
#SBATCH -J cg30b-smoke
#SBATCH -o cg30b-smoke.%j.out
#SBATCH -e cg30b-smoke.%j.err
#SBATCH -p gh-dev
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 02:00:00
#SBATCH -A AST24021

set -euo pipefail

TASK=${TASK:-alfworld}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}
DEFAULT_WORK_BASE=${WORK:-/work/07144/${USER:-$(whoami)}/vista}
if [[ "${SCRATCH:-}" == */vista ]]; then
  DEFAULT_SCRATCH_BASE=$SCRATCH
elif [ -n "${SCRATCH:-}" ]; then
  DEFAULT_SCRATCH_BASE=${SCRATCH}/vista
else
  DEFAULT_SCRATCH_BASE=/scratch/07144/${USER:-$(whoami)}/vista
fi
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ROOT=${CONDA_ROOT:-${DEFAULT_WORK_BASE}/miniconda3}
MODEL_PATH=${MODEL_PATH:-${DEFAULT_WORK_BASE}/models/Qwen3-30B-A3B-Thinking-2507}
HF_HOME=${HF_HOME:-${DEFAULT_WORK_BASE}/hf_cache}
SCRATCH_BASE=${SCRATCH_BASE:-$DEFAULT_SCRATCH_BASE}
ALFWORLD_DATA=${ALFWORLD_DATA:-${SCRATCH_BASE}/alfworld}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-${SCRATCH_BASE}/cache}
SRUN_PARTITION=${SRUN_PARTITION:-gh-dev}
SRUN_TIME=${SRUN_TIME:-02:00:00}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
SAVE_FREQ=${SAVE_FREQ:-1}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-${SCRATCH_BASE}/checkpoints/context-graph-master-smoke}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1}
HF_SYNC_CHECKPOINTS=${HF_SYNC_CHECKPOINTS:-1}
HF_SYNC_REQUIRED=${HF_SYNC_REQUIRED:-1}
HF_NAMESPACE=${HF_NAMESPACE:-lingchensanwen}
HF_REPO_PRIVATE=${HF_REPO_PRIVATE:-false}

export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1
export HF_HOME XDG_CACHE_HOME ALFWORLD_DATA
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
export TORCHDYNAMO_DISABLE=1
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

for env_file in "${WORK:-}/.wandb_env" "${DEFAULT_WORK_BASE}/.wandb_env" "${HOME:-}/.wandb_env"; do
  if [ -n "$env_file" ] && [ -f "$env_file" ]; then
    # shellcheck disable=SC1090
    source "$env_file"
    break
  fi
done
export WANDB_ENTITY=${WANDB_ENTITY:-huancheng}
USE_OPENAI_JUDGE=${USE_OPENAI_JUDGE:-0}
if [ "$TASK" = "hotpotqa" ] && [ "$USE_OPENAI_JUDGE" != "1" ]; then
  export OPENAI_API_KEY=dummy
  unset OPENAI_URL
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate cxtgraph
export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
export PATH="${CONDA_PREFIX}/bin:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}"
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LIBRARY_PATH:-}
export CPATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/include:${CPATH:-}
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$XDG_CACHE_HOME" "$ALFWORLD_DATA"

mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NUM_NODES=${#NODELIST[@]}
if [ "$NUM_NODES" -ne 8 ]; then
  echo "Expected 8 nodes, got $NUM_NODES"
  printf 'Nodes: %s\n' "${NODELIST[*]}"
  exit 1
fi
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
SRUN_PREFIX=(srun --overlap -p "$SRUN_PARTITION" -t "$SRUN_TIME")

if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
else
  TRAINER_LOGGER='["console"]'
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ctxgraph_${TASK}_30b_8n_master_smoke_${TS}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${CHECKPOINT_BASE}/${EXPERIMENT_NAME}}
HF_REPO_ID=${HF_REPO_ID:-${HF_NAMESPACE}/${EXPERIMENT_NAME}}
SEARCH_PID=""
RAY_HEAD_PID=""
WORKER_PIDS=()

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

cleanup() {
  [ -n "$SEARCH_PID" ] && kill "$SEARCH_PID" 2>/dev/null || true
  [ -n "$RAY_HEAD_PID" ] && kill "$RAY_HEAD_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

print_gpu_snapshot() {
  local label="$1"
  echo "--- GPU snapshot: $label ---"
  for node in "${NODELIST[@]}"; do
    "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$node" bash -c '
      printf "%s " "$(hostname -s)"
      nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits || true
    ' || true
  done
}

ray_env='
  source '"${CONDA_ROOT}"'/etc/profile.d/conda.sh
  conda activate cxtgraph
  export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
  export PATH="${CONDA_PREFIX}/bin:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}"
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}
  export HF_HOME='"$HF_HOME"'
  export XDG_CACHE_HOME='"$XDG_CACHE_HOME"'
  export ALFWORLD_DATA='"$ALFWORLD_DATA"'
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export LOCAL_SEARCH_URL='"${LOCAL_SEARCH_URL:-}"'
'
ray_env=${ray_env#$'\n'}
ray_env=${ray_env%$'\n'}
if [ "$TASK" = "hotpotqa" ] && [ "$USE_OPENAI_JUDGE" != "1" ]; then
  ray_env="${ray_env}
  export OPENAI_API_KEY=dummy
  unset OPENAI_URL
"
  ray_env=${ray_env%$'\n'}
fi

echo "=============================================================="
echo "ContextGraph 30B 8-node master smoke"
echo "Task: $TASK"
echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Nodes: $NUM_NODES   Head: $NODE0 ($NODE0_IP)"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Rollout n: $ROLLOUT_N"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Save freq: $SAVE_FREQ"
echo "HF sync: $HF_SYNC_CHECKPOINTS repo=$HF_REPO_ID"
if [ "$TASK" = "hotpotqa" ]; then
  echo "OpenAI judge: ${USE_OPENAI_JUDGE}"
fi
echo "Started: $(date)"
echo "=============================================================="

mkdir -p "$(dirname "$CHECKPOINT_ROOT")"
df -h "$(dirname "$CHECKPOINT_ROOT")" || true

probe "cleanup stale Ray"
for node in "${NODELIST[@]}"; do
  "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$node" bash -c "${ray_env}; ray stop -f >/dev/null 2>&1 || true" || true
done
sleep 5
print_gpu_snapshot "after stale Ray cleanup"

probe "python sanity"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"

if [ "$TASK" = "alfworld" ]; then
  probe "generate ALFWorld hard parquet"
  python scripts/make_alfworld_data.py --hard --n_train 64 --n_val 8
  python - <<'PY'
import pandas as pd
df = pd.read_parquet("data/alfworld_graph_train.parquet")
print("ALFWorld rows:", len(df), "ability:", df["ability"].iloc[0])
PY
elif [ "$TASK" = "hotpotqa" ]; then
  probe "check HotpotQA smoke artifacts"
  for f in data/hotpotqa_graph_train.parquet data/hotpotqa_graph_test.parquet data/hotpotqa_corpus.parquet; do
    [ -f "$f" ] || { echo "Missing $f. Run make_hotpotqa_data.py and build_hotpotqa_corpus.py first."; exit 1; }
  done

  probe "start HotpotQA BM25 search server"
  "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" bash -c "${ray_env}; cd '$PROJECT_ROOT'; export PYTHONPATH='$PROJECT_ROOT':\${PYTHONPATH:-}; exec python -u scripts/hotpotqa_search_server.py --corpus data/hotpotqa_corpus.parquet --port 18999" \
    >/tmp/hp_bm25_${SLURM_JOB_ID:-$$}.log 2>&1 &
  SEARCH_PID=$!

  probe "wait for search server"
  for _ in $(seq 1 120); do
    if curl -fsS "http://${NODE0_IP}:18999/health" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  curl -fsS -X POST -H 'Content-Type: application/json' \
    -d '{"query":"Eiffel Tower","k":1}' \
    "http://${NODE0_IP}:18999/search" >/dev/null || {
      echo "Search server failed. Tail:"
      tail -80 /tmp/hp_bm25_${SLURM_JOB_ID:-$$}.log || true
      exit 1
    }
  export LOCAL_SEARCH_URL="http://${NODE0_IP}:18999"
  probe "search server ready at $LOCAL_SEARCH_URL"
else
  echo "Unknown TASK=$TASK; expected alfworld or hotpotqa"
  exit 1
fi

# Ray workers inherit environment from their `ray start` shells. For HotpotQA,
# make the search endpoint visible there as well as in the trainer process.
ray_env="${ray_env}
  export LOCAL_SEARCH_URL='${LOCAL_SEARCH_URL:-}'
"
ray_env=${ray_env%$'\n'}

probe "start Ray head"
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" bash -c "${ray_env}; ray start --head --node-ip-address='$NODE0_IP' --port=6379 --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block" &
RAY_HEAD_PID=$!
sleep 20

for i in $(seq 1 $((NUM_NODES - 1))); do
  WORKER_NODE=${NODELIST[$i]}
  probe "start Ray worker $WORKER_NODE"
  "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c "${ray_env}; ray start --address='${NODE0_IP}:6379' --num-cpus=70 --num-gpus=1 --block" &
  WORKER_PIDS+=("$!")
  sleep 5
done
sleep 20

export RAY_ADDRESS=${NODE0_IP}:6379
probe "ray status"
ray status || echo "WARN: ray status check failed"
print_gpu_snapshot "before trainer"

COMMON_ARGS=(
  algorithm.adv_estimator=foldgrpo
  algorithm.kl_ctrl.kl_coef=0.001
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.calculate_log_probs=True
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  +actor_rollout_ref.rollout.quantization=fp8
  actor_rollout_ref.rollout.free_cache_engine=False
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False
  actor_rollout_ref.rollout.n=${ROLLOUT_N}
  actor_rollout_ref.rollout.agent.num_workers=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.ref.strategy=fsdp
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.optim.lr=5e-6
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.use_kl_loss=True
  data.return_raw_chat=True
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  +actor_rollout_ref.rollout.plugin.retry_cjk=10
  +actor_rollout_ref.rollout.plugin.enable_summary=False
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005
  +actor_rollout_ref.rollout.plugin.must_finish=False
  +actor_rollout_ref.rollout.plugin.double_check=False
  +actor_rollout_ref.rollout.plugin.must_search=False
  trainer.val_before_train=False
  trainer.val_only=False
  trainer.n_gpus_per_node=1
  trainer.nnodes=${NUM_NODES}
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
  trainer.test_freq=-1
  trainer.save_freq=${SAVE_FREQ}
  trainer.default_local_dir="$CHECKPOINT_ROOT"
  trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
  trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP}
  trainer.project_name=context-graph
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.logger="$TRAINER_LOGGER"
)

if [ "$TASK" = "alfworld" ]; then
  TASK_ARGS=(
    actor_rollout_ref.rollout.prompt_length=16384
    actor_rollout_ref.rollout.response_length=16384
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_num_seqs=64
    data.train_files=data/alfworld_graph_train.parquet
    data.val_files=data/alfworld_graph_test.parquet
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=16384
    data.max_response_length=16384
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=32768
    +actor_rollout_ref.rollout.plugin.workflow=alfworld_graph
    +actor_rollout_ref.rollout.plugin.max_turn=20
    +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=512
    +actor_rollout_ref.rollout.plugin.max_session=4
    +actor_rollout_ref.rollout.plugin.val_max_session=4
    +actor_rollout_ref.rollout.plugin.session_timeout=600
    +actor_rollout_ref.rollout.plugin.branch_len=8192
    +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope,graph]'
    +actor_rollout_ref.rollout.plugin.max_traj=6
    +actor_rollout_ref.rollout.plugin.val_max_turn=20
    +actor_rollout_ref.rollout.plugin.val_response_length=16384
  )
else
  TASK_ARGS=(
    actor_rollout_ref.rollout.prompt_length=4096
    actor_rollout_ref.rollout.response_length=8192
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_num_seqs=64
    data.train_files=data/hotpotqa_graph_train.parquet
    data.val_files=data/hotpotqa_graph_test.parquet
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=4096
    data.max_response_length=8192
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288
    actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=12288
    +actor_rollout_ref.rollout.plugin.workflow=search_graph
    +actor_rollout_ref.rollout.plugin.max_turn=20
    +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=384
    +actor_rollout_ref.rollout.plugin.max_session=3
    +actor_rollout_ref.rollout.plugin.val_max_session=3
    +actor_rollout_ref.rollout.plugin.session_timeout=300
    +actor_rollout_ref.rollout.plugin.branch_len=2048
    +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope,graph]'
    +actor_rollout_ref.rollout.plugin.max_traj=4
    +actor_rollout_ref.rollout.plugin.val_max_turn=20
    +actor_rollout_ref.rollout.plugin.val_response_length=8192
  )
fi

probe "launch trainer"
set +e
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" --chdir="$PROJECT_ROOT" \
  --export=ALL,LOCAL_SEARCH_URL="${LOCAL_SEARCH_URL:-}" \
  python -m scripts.train_graph "${COMMON_ARGS[@]}" "${TASK_ARGS[@]}"
RC=$?
set -e

if [ "$HF_SYNC_CHECKPOINTS" = "1" ] && [ "$SAVE_FREQ" -gt 0 ]; then
  if [ "$RC" -eq 0 ]; then
    probe "sync latest checkpoint to Hugging Face"
    set +e
    bash scripts/sync_latest_checkpoint_to_hf.sh "$CHECKPOINT_ROOT" "$HF_REPO_ID" "$EXPERIMENT_NAME" "$HF_REPO_PRIVATE"
    SYNC_RC=$?
    set -e
    if [ "$SYNC_RC" -ne 0 ]; then
      echo "HF checkpoint sync failed with exit code $SYNC_RC"
      if [ "$HF_SYNC_REQUIRED" = "1" ]; then
        RC=$SYNC_RC
      fi
    fi
  else
    echo "Skipping HF sync because training failed (exit $RC); not uploading partial checkpoints."
  fi
fi

echo "=============================================================="
if [ "$RC" -eq 0 ]; then
  echo "SMOKE PASSED"
else
  echo "SMOKE FAILED (exit $RC)"
fi
echo "Finished: $(date)"
echo "=============================================================="
exit "$RC"
