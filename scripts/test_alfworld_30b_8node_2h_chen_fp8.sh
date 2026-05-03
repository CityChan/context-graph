#!/bin/bash
#SBATCH -J test-alf-30b-chen-fp8-8n
#SBATCH -o test-alf-30b-chen-fp8-8n.%j.out
#SBATCH -e test-alf-30b-chen-fp8-8n.%j.err
#SBATCH -p gh-dev
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 02:00:00
#SBATCH -A AST24021

export WANDB_API_KEY=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

# Chen Apr 19, 2026 setup adapted for Vista idev:
# - BF16 base model
# - vLLM rollout quantization=fp8
# - rollout tensor parallel size = 1
# - FSDP actor/ref
#
# Usage from login node with an existing idev allocation:
#   IDEV_JOBID=<jobid> bash scripts/test_alfworld_30b_8node_2h_chen_fp8.sh
#
# Usage from inside an allocation:
#   bash scripts/test_alfworld_30b_8node_2h_chen_fp8.sh

set -euo pipefail

SRUN_PARTITION=${SRUN_PARTITION:-gh-dev}
SRUN_TIME=${SRUN_TIME:-02:00:00}

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
  echo "Expected 8 nodes for this smoke script, got $NUM_NODES"
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

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Thinking-2507}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')

if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
else
  TRAINER_LOGGER='["console"]'
fi

echo "=============================================================="
echo "Chen FP8 rollout smoke on Vista"
echo "Allocation: $ALLOC_JOB_ID"
echo "Nodes: $NUM_NODES   Head: $NODE0 ($NODE0_IP)"
echo "Model: $MODEL_PATH"
echo "Started: $(date)"
echo "=============================================================="

# ── Pre-download model on head node (single process, avoids NFS race) ──
SNAPSHOT_DIR="$HF_HOME/hub/models--Qwen--Qwen3-30B-A3B-Thinking-2507/snapshots"
NUM_SHARDS=$(find "$SNAPSHOT_DIR" -name "model-*-of-00016.safetensors" 2>/dev/null | wc -l)
if [ "$NUM_SHARDS" -ne 16 ]; then
  echo "--- Model not fully cached ($NUM_SHARDS/16 shards). Downloading on head node ---"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 --cache-dir "$HF_HOME" || {
      echo "Model download failed. Check compute node network access or HF_HOME path."
      exit 1
    }
else
  echo "--- Model already cached: 16/16 shards present ---"
fi

echo "--- Cleaning up any stale Ray processes on allocation nodes ---"
for node in "${NODELIST[@]}"; do
  "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$node" bash -c 'source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh && conda activate cxtgraph && ray stop -f >/dev/null 2>&1 || true' || true
done
sleep 5

python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
python -c "import textworld, alfworld; print('textworld + alfworld OK')"

echo "--- Generating ALFWorld parquet ---"
python scripts/make_alfworld_data.py --n_train 32 --n_val 8

echo "--- Starting Ray head on $NODE0 ---"
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" bash -c '
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  hash -r
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
  export HF_HOME='"$HF_HOME"'
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
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    hash -r
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
    export HF_HOME='"$HF_HOME"'
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

echo "=============================================================="
echo "Launching Chen FP8 rollout smoke test on 8 nodes"
echo "=============================================================="

set +e
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" --chdir="$PROJECT_ROOT" python -m scripts.train_fold \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=4096 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False \
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
  actor_rollout_ref.actor.use_kl_loss=True \
  data.train_files=data/alfworld_train.parquet \
  data.val_files=data/alfworld_test.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=4096 \
  data.max_response_length=8192 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=12288 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld \
  +actor_rollout_ref.rollout.plugin.max_turn=20 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=512 \
  +actor_rollout_ref.rollout.plugin.max_session=3 \
  +actor_rollout_ref.rollout.plugin.val_max_session=3 \
  +actor_rollout_ref.rollout.plugin.session_timeout=300 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=2048 \
  +actor_rollout_ref.rollout.plugin.process_reward=none \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=20 \
  +actor_rollout_ref.rollout.plugin.val_response_length=8192 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=3 \
  trainer.test_freq=999 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.experiment_name=chen_fp8_rollout_smoke_30b_8n \
  trainer.logger="$TRAINER_LOGGER"
RC=$?
set -e

echo "=============================================================="
if [ $RC -eq 0 ]; then
  echo "SMOKE TEST PASSED"
else
  echo "SMOKE TEST FAILED (exit $RC)"
fi
echo "Finished: $(date)"
echo "=============================================================="

exit $RC
