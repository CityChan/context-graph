#!/bin/bash

# One real full-parameter ContextGraph SFT optimizer step for the original
# zero-shot Qwen/Qwen3-8B checkpoint in an existing four-node Vista idev
# allocation. This is a mechanics smoke, not the formal SFT run.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"

# Never inherit a MODEL_PATH left over from another model evaluation. Prefer
# an explicit QWEN3_8B_MODEL_PATH; otherwise reuse the read-only Qwen3-8B cache
# that supplied the successful zero-shot evaluations.
unset MODEL_PATH
if [ -n "${QWEN3_8B_MODEL_PATH:-}" ]; then
  export MODEL_PATH="$QWEN3_8B_MODEL_PATH"
else
  export WORK_MODEL_CACHE_ROOT=${WORK_MODEL_CACHE_ROOT:-/work/09281/chc_1996/vista/cache/hub/models--Qwen--Qwen3-8B/snapshots}
  MODEL_PATH=$(find "$WORK_MODEL_CACHE_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1) || true
  if [ -z "${MODEL_PATH:-}" ] || [ ! -s "$MODEL_PATH/config.json" ]; then
    echo "ERROR: no Qwen3-8B snapshot found under $WORK_MODEL_CACHE_ROOT"
    echo "Set QWEN3_8B_MODEL_PATH to a complete snapshot directory."
    exit 2
  fi
  export MODEL_PATH
  export ALLOW_WORK_MODEL_CACHE=1
fi

export MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
export TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-cxtgraph}
export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
export TRAIN_FILE=${TRAIN_FILE:-$SCRATCH/contextgraph_sft/deepseek_v4_flash_0731_interactive/945161_0/alfworld/contextgraph_sft_train.parquet}
export VAL_FILE=${VAL_FILE:-$SCRATCH/contextgraph_sft/deepseek_v4_flash_0731_interactive/945161_0/alfworld/contextgraph_sft_validation.parquet}
export MAX_LENGTH=${MAX_LENGTH:-8192}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-4}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-4}
export LORA_RANK=${LORA_RANK:-0}
export LORA_ALPHA=${LORA_ALPHA:-16}
export TRAIN_LR=${TRAIN_LR:-1e-5}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export LOSS_MASK_MODE=chatml
export PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
export DATA_PREFLIGHT_TIMEOUT=${DATA_PREFLIGHT_TIMEOUT:-600}
export MASTER_PORT=${MASTER_PORT:-29527}
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}_qwen3_8b_sft_fullparam_smoke}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH/contextgraph_sft_checkpoints/$RUN_TAG}

exec bash "$PROJECT_ROOT/scripts/smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh"
