#!/bin/bash
#SBATCH -J cg30b-compare
#SBATCH -o logs/cg30b-compare.%j.out
#SBATCH -e logs/cg30b-compare.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A AST24021

set -euo pipefail

# TASK:  alfworld
# AGENT: ctxgraph | baseline
TASK=${TASK:-alfworld}
AGENT=${AGENT:-ctxgraph}
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
SRUN_PARTITION=${SRUN_PARTITION:-gh}
SRUN_TIME=${SRUN_TIME:-12:00:00}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-${SCRATCH_BASE}/checkpoints/context-graph-compare}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-2}

if [ "$TASK" != "alfworld" ]; then
  echo "Unknown TASK=$TASK; expected alfworld"
  exit 2
fi
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-16}
SAVE_FREQ=${SAVE_FREQ:-4}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-16384}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-16384}
ROLLOUT_LOG_PROB_MAX_LEN=${ROLLOUT_LOG_PROB_MAX_LEN:-32768}
DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-16384}
DATA_MAX_RESPONSE_LENGTH=${DATA_MAX_RESPONSE_LENGTH:-16384}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_TURN=${MAX_TURN:-20}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-512}
MAX_SESSION=${MAX_SESSION:-4}
BRANCH_LEN=${BRANCH_LEN:-8192}
MAX_TRAJ=${MAX_TRAJ:-6}

case "$AGENT" in
  ctxgraph|baseline) ;;
  *) echo "Unknown AGENT=$AGENT; expected ctxgraph or baseline"; exit 2 ;;
esac

ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-${ROLLOUT_LOG_PROB_MAX_LEN}}
ACTOR_PPO_INFER_MAX_TOKEN_LEN=${ACTOR_PPO_INFER_MAX_TOKEN_LEN:-${ROLLOUT_LOG_PROB_MAX_LEN}}

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
mkdir -p "$XDG_CACHE_HOME" "$ALFWORLD_DATA" "$CHECKPOINT_BASE"

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
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${AGENT}_${TASK}_30b_8n_compare_${TS}}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${CHECKPOINT_BASE}/${EXPERIMENT_NAME}}
RAY_HEAD_PID=""
WORKER_PIDS=()

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

cleanup() {
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
'
ray_env=${ray_env#$'\n'}
ray_env=${ray_env%$'\n'}

echo "=============================================================="
echo "30B 8-node comparison run"
echo "Task: $TASK"
echo "Agent: $AGENT"
echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Nodes: $NUM_NODES   Head: $NODE0 ($NODE0_IP)"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Steps/save_freq: $TOTAL_TRAINING_STEPS/$SAVE_FREQ"
echo "Started: $(date)"
echo "=============================================================="
df -h "$CHECKPOINT_BASE" || true

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

probe "generate ALFWorld hard parquet"
python scripts/make_alfworld_data.py --hard --n_train "${ALFWORLD_N_TRAIN:-1000}" --n_val "${ALFWORLD_N_VAL:-100}"
if [ "$AGENT" = "ctxgraph" ]; then
  DATA_TRAIN=data/alfworld_graph_train.parquet
  DATA_VAL=data/alfworld_graph_test.parquet
  WORKFLOW=alfworld_graph
  DEFAULT_AGENT_LOOP=context_graph_isolated_agent
  PROCESS_REWARD='[flat,scope,graph]'
else
  DATA_TRAIN=data/alfworld_train.parquet
  DATA_VAL=data/alfworld_test.parquet
  WORKFLOW=alfworld_branch
  DEFAULT_AGENT_LOOP=fold_agent
  PROCESS_REWARD='[flat,scope]'
fi
python - <<PY
import pandas as pd
df = pd.read_parquet("$DATA_TRAIN")
assert df["ability"].iloc[0] == "ALFWorld@hard", df["ability"].iloc[0]
print("ALFWorld data ok:", "$DATA_TRAIN", len(df), df["ability"].iloc[0])
PY

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

probe "launch trainer"
set +e
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" --chdir="$PROJECT_ROOT" \
  --export=ALL,LOCAL_SEARCH_URL="${LOCAL_SEARCH_URL:-}" \
  python -m scripts.train_graph \
    algorithm.adv_estimator=foldgrpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    actor_rollout_ref.rollout.agent.default_agent_loop="$DEFAULT_AGENT_LOOP" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.response_length=${ROLLOUT_RESPONSE_LENGTH} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_LEN} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.quantization=fp8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    data.train_files="$DATA_TRAIN" \
    data.val_files="$DATA_VAL" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${DATA_MAX_PROMPT_LENGTH} \
    data.max_response_length=${DATA_MAX_RESPONSE_LENGTH} \
    data.return_raw_chat=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN} \
    actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${ACTOR_PPO_INFER_MAX_TOKEN_LEN} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.rollout.plugin.workflow="$WORKFLOW" \
    +actor_rollout_ref.rollout.plugin.max_turn=${MAX_TURN} \
    +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
    +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=${TURN_MAX_NEW_TOKENS} \
    +actor_rollout_ref.rollout.plugin.max_session=${MAX_SESSION} \
    +actor_rollout_ref.rollout.plugin.val_max_session=${MAX_SESSION} \
    +actor_rollout_ref.rollout.plugin.session_timeout=600 \
    +actor_rollout_ref.rollout.plugin.enable_summary=False \
    +actor_rollout_ref.rollout.plugin.branch_len=${BRANCH_LEN} \
    +actor_rollout_ref.rollout.plugin.process_reward="$PROCESS_REWARD" \
    +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
    +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
    +actor_rollout_ref.rollout.plugin.max_traj=${MAX_TRAJ} \
    +actor_rollout_ref.rollout.plugin.must_finish=False \
    +actor_rollout_ref.rollout.plugin.double_check=False \
    +actor_rollout_ref.rollout.plugin.must_search=False \
    +actor_rollout_ref.rollout.plugin.val_max_turn=${MAX_TURN} \
    +actor_rollout_ref.rollout.plugin.val_response_length=${ROLLOUT_RESPONSE_LENGTH} \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=${NUM_NODES} \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.test_freq=-1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.default_local_dir="$CHECKPOINT_ROOT" \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP} \
    trainer.project_name=context-graph \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger="$TRAINER_LOGGER"
RC=$?
set -e
print_gpu_snapshot "after trainer"

echo "=============================================================="
if [ "$RC" -eq 0 ]; then
  echo "RUN COMPLETED"
else
  echo "RUN FAILED (exit $RC)"
fi
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Finished: $(date)"
echo "=============================================================="
exit "$RC"
