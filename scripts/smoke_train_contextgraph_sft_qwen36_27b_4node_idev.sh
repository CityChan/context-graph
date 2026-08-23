#!/bin/bash

# One real multi-turn SFT optimizer step for Qwen3.6-27B inside an existing
# four-node Vista idev allocation. All four GH200s train one repeated smoke
# sample with FSDP2 data parallelism and PyTorch SDPA. The default LoRA smoke
# saves a real sharded checkpoint while avoiding an external flash-attn build.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-27B}
TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-deepseek_v4}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
MAX_LENGTH=${MAX_LENGTH:-8192}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
TRAIN_LR=${TRAIN_LR:-1e-5}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
MASTER_PORT=${MASTER_PORT:-29517}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}_qwen36_27b_sft_smoke}

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
# Use SFT-specific overrides so an inherited login-shell HF_HOME under /work
# cannot silently select the shared model cache. JIT/model reads on Vista must
# use SCRATCH for this training path.
HF_HOME=${SFT_HF_HOME:-$SCRATCH/hf_cache}
HF_HUB_CACHE=${SFT_HF_HUB_CACHE:-$HF_HOME/hub}
DATA_ROOT=${DATA_ROOT:-$SCRATCH/contextgraph_sft/deepseek_v4_flash_0731_interactive}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH/contextgraph_sft_checkpoints/$RUN_TAG}

resolve_snapshot() {
  local repo_id=$1
  local cache_name="models--${repo_id//\//--}"
  local snapshot
  snapshot=$(find "$HF_HUB_CACHE/$cache_name/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
  if [ -n "$snapshot" ] && [ -s "$snapshot/config.json" ]; then
    printf '%s\n' "$snapshot"
    return 0
  fi
  return 1
}

if [ -n "${MODEL_PATH:-}" ]; then
  if [ ! -s "$MODEL_PATH/config.json" ]; then
    echo "ERROR: MODEL_PATH does not contain config.json: $MODEL_PATH"
    exit 2
  fi
else
  MODEL_PATH=$(resolve_snapshot "$MODEL_ID") || true
fi
if [ -z "${MODEL_PATH:-}" ]; then
  echo "ERROR: $MODEL_ID is not cached under $HF_HUB_CACHE"
  echo "Set MODEL_PATH to its snapshot directory under SCRATCH."
  exit 2
fi

if [ -z "${TRAIN_FILE:-}" ]; then
  TRAIN_FILE=$(find "$DATA_ROOT" -type f -name contextgraph_sft_train.parquet -size +0c -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-) || true
fi
if [ -z "${TRAIN_FILE:-}" ] || [ ! -s "$TRAIN_FILE" ]; then
  echo "ERROR: no non-empty ContextGraph SFT parquet found under $DATA_ROOT"
  echo "Set TRAIN_FILE=/absolute/path/to/contextgraph_sft_train.parquet."
  exit 2
fi

require_scratch_path() {
  local name=$1
  local value=$2
  case "$value" in
    "$SCRATCH"/*) ;;
    *) echo "ERROR: $name must be stored under SCRATCH=$SCRATCH, got $value"; exit 2 ;;
  esac
}
require_scratch_path MODEL_PATH "$MODEL_PATH"
require_scratch_path HF_HOME "$HF_HOME"
require_scratch_path HF_HUB_CACHE "$HF_HUB_CACHE"
require_scratch_path CHECKPOINT_ROOT "$CHECKPOINT_ROOT"

activate_train_env() {
  set +u
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate "$TRAIN_CONDA_ENV"
  set -u
  export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8
  export CUDA_MATH_ROOT=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8
  export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/bin:$PATH"
  export CC=gcc
  export CXX=g++
  export CUDAHOSTCXX=g++
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export CPATH="$CUDA_MATH_ROOT/targets/sbsa-linux/include:$CUDA_HOME/include:${CPATH:-}"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_MATH_ROOT/targets/sbsa-linux/lib:$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="$CUDA_MATH_ROOT/targets/sbsa-linux/lib:$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
  export HF_HOME HF_HUB_CACHE
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_FILE_LOCKING=1
  export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  export TOKENIZERS_PARALLELISM=false
  export HYDRA_FULL_ERROR=1
  export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
  export TORCHINDUCTOR_CACHE_DIR="/tmp/contextgraph-sft-inductor-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}"
  export TRITON_CACHE_DIR="/tmp/contextgraph-sft-triton-${SLURM_JOB_ID:-local}-${SLURM_PROCID:-0}"
}

SCRIPT_PATH=$(readlink -f "$0")

if [ "${SFT_PREFLIGHT_WORKER:-0}" = "1" ]; then
  activate_train_env
  cd "$PROJECT_ROOT"
  python -c "import accelerate, codetiming, hydra, omegaconf, pandas, peft, pyarrow, tensordict, torch, torchdata, transformers; from transformers import AutoConfig; c=AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True, local_files_only=True); print('node preflight:', 'host='+__import__('socket').gethostname(), 'torch='+torch.__version__, 'transformers='+transformers.__version__, 'model_type='+str(getattr(c, 'model_type', None)))"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  exit 0
fi

if [ "${SFT_TRAIN_WORKER:-0}" = "1" ]; then
  activate_train_env
  cd "$PROJECT_ROOT"
  exec torchrun --nnodes="$NUM_NODES" --nproc-per-node=1 --node-rank="$SLURM_PROCID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$TRAIN_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    data.train_max_samples="$NUM_NODES" \
    data.val_max_samples="$NUM_NODES" \
    data.multiturn.enable=True \
    data.max_length="$MAX_LENGTH" \
    data.truncation=right \
    model.partial_pretrain="$MODEL_PATH" \
    model.trust_remote_code=True \
    model.attn_implementation="$ATTN_IMPLEMENTATION" \
    model.fsdp_config.model_dtype=bfloat16 \
    model.enable_gradient_checkpointing=True \
    model.lora_rank="$LORA_RANK" \
    model.lora_alpha="$LORA_ALPHA" \
    model.target_modules=all-linear \
    model.strategy=fsdp2 \
    optim.lr="$TRAIN_LR" \
    optim.lr_warmup_steps_ratio=0.0 \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=False \
    trainer.project_name=contextgraph-sft \
    trainer.experiment_name="$RUN_TAG" \
    trainer.default_local_dir="$CHECKPOINT_ROOT" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.logger='["console"]' \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.nnodes="$NUM_NODES" \
    trainer.n_gpus_per_node=1 \
    trainer.resume_mode=disable \
    trainer.max_ckpt_to_keep=1 \
    'trainer.checkpoint.save_contents=["model","extra"]' \
    'trainer.checkpoint.load_contents=["model","extra"]'
fi

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this script inside an active Vista idev/Slurm allocation"
  exit 2
fi
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NUM_NODES=${#NODELIST[@]}
if [ "$NUM_NODES" -ne "$EXPECTED_NUM_NODES" ]; then
  echo "ERROR: expected $EXPECTED_NUM_NODES allocated nodes, got $NUM_NODES"
  exit 2
fi
MASTER_ADDR=$(getent hosts "${NODELIST[0]}" | awk '{print $1}')
TRAIN_FILES="[$TRAIN_FILE,$TRAIN_FILE,$TRAIN_FILE,$TRAIN_FILE]"
export MODEL_PATH TRAIN_FILE TRAIN_FILES CHECKPOINT_ROOT NUM_NODES MASTER_ADDR MASTER_PORT ATTN_IMPLEMENTATION

mkdir -p "$PROJECT_ROOT/logs" "$CHECKPOINT_ROOT"
cd "$PROJECT_ROOT"
activate_train_env

echo "Qwen3.6-27B ContextGraph SFT training smoke"
echo "Model: $MODEL_PATH"
echo "Data: $TRAIN_FILE"
echo "Nodes: ${NODELIST[*]}"
echo "Parallelism: FSDP2 world=$NUM_NODES, Ulysses SP=1, DP=$NUM_NODES"
echo "Training: steps=$TOTAL_TRAINING_STEPS max_length=$MAX_LENGTH LoRA rank=$LORA_RANK attention=$ATTN_IMPLEMENTATION"
echo "Checkpoint: $CHECKPOINT_ROOT"

python scripts/check_contextgraph_sft_data.py --data "$TRAIN_FILE" --tokenizer "$MODEL_PATH" --max-length "$MAX_LENGTH"
echo "Preflight: checking the training stack and GPU memory on all nodes"
srun --overlap --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 env SFT_PREFLIGHT_WORKER=1 bash "$SCRIPT_PATH"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "Qwen3.6-27B ContextGraph SFT training preflight passed."
  exit 0
fi

echo "Launching one real SFT optimizer step"
srun --overlap --nodes="$NUM_NODES" --ntasks="$NUM_NODES" --ntasks-per-node=1 env SFT_TRAIN_WORKER=1 bash "$SCRIPT_PATH"

STEP_DIR="$CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS"
if [ ! -s "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]; then
  echo "ERROR: training completed without a checkpoint tracker"
  exit 3
fi
if [ ! -d "$STEP_DIR" ]; then
  echo "ERROR: expected checkpoint directory was not created: $STEP_DIR"
  exit 3
fi
MODEL_SHARDS=$(find "$STEP_DIR" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' -size +0c | wc -l)
if [ "$MODEL_SHARDS" -ne "$NUM_NODES" ]; then
  echo "ERROR: expected $NUM_NODES non-empty model shards, found $MODEL_SHARDS"
  exit 3
fi

echo "Qwen3.6-27B ContextGraph SFT training smoke passed: checkpoint=$STEP_DIR shards=$MODEL_SHARDS"
