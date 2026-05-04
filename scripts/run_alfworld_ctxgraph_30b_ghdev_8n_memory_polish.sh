#!/bin/bash
#SBATCH -J ctxgraph-alf-30b-8n-mempolish
#SBATCH -o ctxgraph-alf-30b-8n-mempolish.%j.out
#SBATCH -e ctxgraph-alf-30b-8n-mempolish.%j.err
#SBATCH -p gh-dev
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 00:30:00
#SBATCH -A AST24021

# Short 8-node Context-Graph ALFWorld memory-polish run for Vista gh-dev.
# It keeps the BF16 actor + FP8 rollout + ContextGraph workflow, while exposing
# the vLLM memory knobs that determine whether a second wake-up can fit.

set -euo pipefail

SRUN_PARTITION=${SRUN_PARTITION:-gh-dev}
SRUN_TIME=${SRUN_TIME:-00:30:00}

if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
  ALLOC_JOB_ID="$SLURM_JOB_ID"
  NODELIST_SPEC="$SLURM_JOB_NODELIST"
  SRUN_PREFIX=(srun --overlap -p "$SRUN_PARTITION" -t "$SRUN_TIME")
elif [ -n "${IDEV_JOBID:-}" ]; then
  ALLOC_JOB_ID="$IDEV_JOBID"
  NODELIST_SPEC=$(squeue -j "$ALLOC_JOB_ID" -h -o "%N" | head -n 1)
  if [ -z "$NODELIST_SPEC" ]; then
    echo "Could not resolve nodes for IDEV_JOBID=$ALLOC_JOB_ID"
    exit 1
  fi
  SRUN_PREFIX=(srun --jobid="$ALLOC_JOB_ID" --overlap -p "$SRUN_PARTITION" -t "$SRUN_TIME")
else
  echo "Run inside a Slurm allocation or set IDEV_JOBID=<jobid>."
  exit 1
fi

mapfile -t NODELIST < <(scontrol show hostnames "$NODELIST_SPEC")
NUM_NODES=${#NODELIST[@]}
if [ "$NUM_NODES" -ne 8 ]; then
  echo "Expected 8 nodes for this script, got $NUM_NODES"
  printf 'Nodes: %s\n' "${NODELIST[*]}"
  exit 1
fi

export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

if [ -n "${WORK:-}" ] && [ -f "$WORK/.wandb_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.wandb_env"
fi

source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh
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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PROJECT_ROOT=/work/07144/yw23374/vista/context-graph
MODEL_PATH=${MODEL_PATH:-/work/07144/yw23374/vista/models/Qwen3-30B-A3B-Thinking-2507}
export HF_HOME=${HF_HOME:-/work/07144/yw23374/vista/hf_cache}
export VISTA_SCRATCH_BASE=${VISTA_SCRATCH_BASE:-/scratch/07144/yw23374/vista}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${VISTA_SCRATCH_BASE}/cache}
export ALFWORLD_DATA=${ALFWORLD_DATA:-${VISTA_SCRATCH_BASE}/alfworld}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

export WANDB_ENTITY=${WANDB_ENTITY:-huancheng}

NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')

if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
else
  TRAINER_LOGGER='["console"]'
fi

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ctxgraph_alfworld_30b_8n_mempolish}
DATA_N_TRAIN=${DATA_N_TRAIN:-64}
DATA_N_VAL=${DATA_N_VAL:-16}
SAVE_FREQ=${SAVE_FREQ:-1}
TEST_FREQ=${TEST_FREQ:--1}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-/scratch/07144/yw23374/vista/checkpoints/context-graph}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${CHECKPOINT_BASE}/${EXPERIMENT_NAME}}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-2}
HF_SYNC_CHECKPOINTS=${HF_SYNC_CHECKPOINTS:-1}
HF_SYNC_REQUIRED=${HF_SYNC_REQUIRED:-1}
HF_NAMESPACE=${HF_NAMESPACE:-lingchensanwen}
HF_REPO_ID=${HF_REPO_ID:-${HF_NAMESPACE}/${EXPERIMENT_NAME}}
HF_REPO_PRIVATE=${HF_REPO_PRIVATE:-false}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-6144}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
ROLLOUT_ENABLE_SLEEP_MODE=${ROLLOUT_ENABLE_SLEEP_MODE:-False}
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-4096}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-2048}
ROLLOUT_LOG_PROB_MAX_LEN=${ROLLOUT_LOG_PROB_MAX_LEN:-$((ROLLOUT_PROMPT_LENGTH + ROLLOUT_RESPONSE_LENGTH))}
ROLLOUT_N=${ROLLOUT_N:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
DATA_MAX_PROMPT_LENGTH=${DATA_MAX_PROMPT_LENGTH:-${ROLLOUT_PROMPT_LENGTH}}
DATA_MAX_RESPONSE_LENGTH=${DATA_MAX_RESPONSE_LENGTH:-${ROLLOUT_RESPONSE_LENGTH}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
ACTOR_PPO_MAX_TOKEN_LEN=${ACTOR_PPO_MAX_TOKEN_LEN:-${ROLLOUT_LOG_PROB_MAX_LEN}}
ACTOR_PPO_INFER_MAX_TOKEN_LEN=${ACTOR_PPO_INFER_MAX_TOKEN_LEN:-${ROLLOUT_LOG_PROB_MAX_LEN}}
PLUGIN_MAX_TURN=${PLUGIN_MAX_TURN:-8}
PLUGIN_TURN_MAX_NEW_TOKENS=${PLUGIN_TURN_MAX_NEW_TOKENS:-128}
PLUGIN_MAX_SESSION=${PLUGIN_MAX_SESSION:-2}
PLUGIN_VAL_MAX_SESSION=${PLUGIN_VAL_MAX_SESSION:-2}
PLUGIN_SESSION_TIMEOUT=${PLUGIN_SESSION_TIMEOUT:-600}
PLUGIN_BRANCH_LEN=${PLUGIN_BRANCH_LEN:-1024}
PLUGIN_MAX_TRAJ=${PLUGIN_MAX_TRAJ:-3}
PLUGIN_VAL_MAX_TURN=${PLUGIN_VAL_MAX_TURN:-${PLUGIN_MAX_TURN}}
PLUGIN_VAL_RESPONSE_LENGTH=${PLUGIN_VAL_RESPONSE_LENGTH:-${ROLLOUT_RESPONSE_LENGTH}}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}

echo "=============================================================="
echo "Reduced Context-Graph FP8 run on Vista"
echo "Allocation: $ALLOC_JOB_ID"
echo "Nodes: $NUM_NODES   Head: $NODE0 ($NODE0_IP)"
echo "Model: $MODEL_PATH"
echo "Steps: $TOTAL_TRAINING_STEPS"
echo "Save freq: $SAVE_FREQ"
echo "Test freq: $TEST_FREQ"
echo "Rollout memory: gpu_util=${ROLLOUT_GPU_MEMORY_UTILIZATION} eager=${ROLLOUT_ENFORCE_EAGER} max_seqs=${ROLLOUT_MAX_NUM_SEQS} max_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
echo "Rollout length: prompt=${ROLLOUT_PROMPT_LENGTH} response=${ROLLOUT_RESPONSE_LENGTH} logprob_max=${ROLLOUT_LOG_PROB_MAX_LEN} n=${ROLLOUT_N}"
echo "Batch: train=${TRAIN_BATCH_SIZE} ppo_mini=${PPO_MINI_BATCH_SIZE} actor_token_max=${ACTOR_PPO_MAX_TOKEN_LEN}"
echo "Plugin: max_turn=${PLUGIN_MAX_TURN} turn_tokens=${PLUGIN_TURN_MAX_NEW_TOKENS} max_session=${PLUGIN_MAX_SESSION} branch_len=${PLUGIN_BRANCH_LEN} max_traj=${PLUGIN_MAX_TRAJ}"
echo "Actor offload: param=${ACTOR_PARAM_OFFLOAD} optimizer=${ACTOR_OPTIMIZER_OFFLOAD}"
echo "Rollout sleep/free-cache: free_cache=${ROLLOUT_FREE_CACHE_ENGINE} sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE}"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Checkpoint retention: actor=${MAX_ACTOR_CKPT_TO_KEEP} critic=${MAX_CRITIC_CKPT_TO_KEEP}"
echo "ALFWORLD_DATA: ${ALFWORLD_DATA}"
echo "WANDB_ENTITY: ${WANDB_ENTITY}"
echo "HF sync: ${HF_SYNC_CHECKPOINTS} repo=${HF_REPO_ID}"
echo "Started: $(date)"
echo "=============================================================="

mkdir -p "$XDG_CACHE_HOME" "$ALFWORLD_DATA"
mkdir -p "$(dirname "$CHECKPOINT_ROOT")"
df -h "$(dirname "$CHECKPOINT_ROOT")" || true
lfs quota -h -u "${USER:-$(whoami)}" "$(dirname "$CHECKPOINT_ROOT")" 2>/dev/null || true

print_gpu_snapshot() {
  local label="$1"
  echo "--- GPU snapshot: ${label} ---"
  for node in "${NODELIST[@]}"; do
    "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$node" bash -c '
      printf "%s " "$(hostname -s)"
      nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory \
        --format=csv,noheader,nounits || true
    ' || true
  done
}

if [ "$HF_SYNC_CHECKPOINTS" = "1" ] && [ "$SAVE_FREQ" -gt 0 ]; then
  echo "--- Hugging Face preflight ---"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf auth whoami
fi

echo "--- Cleaning up any stale Ray processes on allocation nodes ---"
for node in "${NODELIST[@]}"; do
  "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$node" bash -c 'source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh && conda activate cxtgraph && ray stop -f >/dev/null 2>&1 || true' || true
done
sleep 5
print_gpu_snapshot "after stale Ray cleanup"

python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
python -c "import textworld, alfworld; print('textworld + alfworld OK')"

echo "--- Generating ALFWorld parquet ---"
python scripts/make_alfworld_data.py --n_train "$DATA_N_TRAIN" --n_val "$DATA_N_VAL"

echo "--- Starting Ray head on $NODE0 ---"
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" bash -c '
  source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  hash -r
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
  export HF_HOME='"$HF_HOME"'
  export XDG_CACHE_HOME='"$XDG_CACHE_HOME"'
  export ALFWORLD_DATA='"$ALFWORLD_DATA"'
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  ray start --head --node-ip-address='"$NODE0_IP"' --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
' &
RAY_HEAD_PID=$!
sleep 20

WORKER_PIDS=()
for i in $(seq 1 $((NUM_NODES - 1))); do
  WORKER_NODE=${NODELIST[$i]}
  echo "--- Starting Ray worker on $WORKER_NODE (node $i) ---"
  "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c '
    source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    hash -r
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
    export HF_HOME='"$HF_HOME"'
    export XDG_CACHE_HOME='"$XDG_CACHE_HOME"'
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

cleanup() {
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

export RAY_ADDRESS=${NODE0_IP}:6379
echo "--- Ray cluster status ---"
ray status || echo "WARN: ray status check failed"
print_gpu_snapshot "before trainer launch"

echo "=============================================================="
echo "Launching Context-Graph FP8 checkpoint/HF sync validation on 8 nodes"
echo "=============================================================="

set +e
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" --chdir="$PROJECT_ROOT" python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${ROLLOUT_RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_LEN} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
  actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
  actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS} \
  actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
  actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE} \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=${ROLLOUT_ENABLE_SLEEP_MODE} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  data.train_files=data/alfworld_graph_train.parquet \
  data.val_files=data/alfworld_graph_test.parquet \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${DATA_MAX_PROMPT_LENGTH} \
  data.max_response_length=${DATA_MAX_RESPONSE_LENGTH} \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN} \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${ACTOR_PPO_INFER_MAX_TOKEN_LEN} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld_graph \
  +actor_rollout_ref.rollout.plugin.max_turn=${PLUGIN_MAX_TURN} \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=${PLUGIN_TURN_MAX_NEW_TOKENS} \
  +actor_rollout_ref.rollout.plugin.max_session=${PLUGIN_MAX_SESSION} \
  +actor_rollout_ref.rollout.plugin.val_max_session=${PLUGIN_VAL_MAX_SESSION} \
  +actor_rollout_ref.rollout.plugin.session_timeout=${PLUGIN_SESSION_TIMEOUT} \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=${PLUGIN_BRANCH_LEN} \
  +actor_rollout_ref.rollout.plugin.process_reward='"'"'[flat,scope,graph]'"'"' \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
  +actor_rollout_ref.rollout.plugin.max_traj=${PLUGIN_MAX_TRAJ} \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=${PLUGIN_VAL_MAX_TURN} \
  +actor_rollout_ref.rollout.plugin.val_response_length=${PLUGIN_VAL_RESPONSE_LENGTH} \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.default_local_dir="${CHECKPOINT_ROOT}" \
  trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
  trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP} \
  trainer.project_name=context-graph \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.logger="$TRAINER_LOGGER"
RC=$?
set -e

if [ "$HF_SYNC_CHECKPOINTS" = "1" ] && [ "$SAVE_FREQ" -gt 0 ]; then
  if [ "$RC" -eq 0 ]; then
    echo "--- Syncing latest checkpoint to Hugging Face ---"
    set +e
    bash scripts/sync_latest_checkpoint_to_hf.sh "$CHECKPOINT_ROOT" "$HF_REPO_ID" "$EXPERIMENT_NAME" "$HF_REPO_PRIVATE"
    SYNC_RC=$?
    set -e
    if [ "$SYNC_RC" -ne 0 ]; then
      echo "HF checkpoint sync failed with exit code $SYNC_RC"
      if [ "$RC" -eq 0 ] && [ "$HF_SYNC_REQUIRED" = "1" ]; then
        RC=$SYNC_RC
      fi
    fi
  else
    echo "Skipping HF sync because training failed (exit $RC); not uploading partial checkpoints."
  fi
fi

echo "=============================================================="
if [ $RC -eq 0 ]; then
  echo "RUN COMPLETED"
else
  echo "RUN FAILED (exit $RC)"
fi
echo "Finished: $(date)"
echo "=============================================================="

exit $RC
