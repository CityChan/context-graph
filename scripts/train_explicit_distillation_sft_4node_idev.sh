#!/bin/bash
# Context distillation, not RL or exact replay. Default: one-step mechanics smoke.
set -euo pipefail
export PROJECT_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
: "${EXPLICIT_SFT_DIR:?Set the exported explicit distillation directory}"
: "${SFT_STUDENT_MODEL_PATH:?Set the original student HF checkpoint, not the RL teacher}"
: "${HELDOUT_DATA:?Set the actual independent evaluation parquet used at collection}"
export MODEL_ID=${SFT_STUDENT_MODEL_ID:-Qwen/Qwen3.5-9B} MODEL_PATH=$SFT_STUDENT_MODEL_PATH
export TRAIN_FILE=$EXPLICIT_SFT_DIR/train.parquet VAL_FILE=$EXPLICIT_SFT_DIR/validation.parquet
export SFT_EXPLICIT_EXPORT=$EXPLICIT_SFT_DIR SFT_EXPLICIT_HELDOUT=$HELDOUT_DATA
export LOSS_MASK_MODE=agent_context SFT_TRUNCATION=error DATA_PREFLIGHT_ALL=1
export TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-deepseek_v4} EXPECTED_NUM_NODES=4
export MAX_LENGTH=${MAX_LENGTH:-65536} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 LORA_RANK=${LORA_RANK:-0} TRAIN_LR=${TRAIN_LR:-1e-5}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}_explicit_distillation}
echo "EXPLICIT_DISTILLATION_TRAIN context=$MAX_LENGTH steps=$TOTAL_TRAINING_STEPS student=$MODEL_PATH export=$EXPLICIT_SFT_DIR"
exec bash "$PROJECT_ROOT/scripts/smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh"
