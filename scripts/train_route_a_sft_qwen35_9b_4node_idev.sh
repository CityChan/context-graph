#!/bin/bash
# Exact native-token SFT; default one-step mechanics smoke, explicit longer runs.
set -euo pipefail
export PROJECT_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
: "${ROUTE_A_SFT_DIR:?Set ROUTE_A_SFT_DIR to the exported Route A sft-* directory}"
: "${QWEN35_9B_MODEL_PATH:?Use the same local checkpoint used for Route A collection}"
export MODEL_ID=Qwen/Qwen3.5-9B MODEL_PATH=$QWEN35_9B_MODEL_PATH
export TRAIN_FILE=$ROUTE_A_SFT_DIR/train.parquet VAL_FILE=$ROUTE_A_SFT_DIR/validation.parquet
[[ -s "$TRAIN_FILE" && -s "$VAL_FILE" ]] || { echo 'Need nonempty train and validation exports; collect more tasks without changing the split' >&2; exit 2; }
export LOSS_MASK_MODE=agent_context SFT_TRUNCATION=error DATA_PREFLIGHT_ALL=1
export TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-deepseek_v4} EXPECTED_NUM_NODES=4
export MAX_LENGTH=${MAX_LENGTH:-65536} TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4} MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1
export LORA_RANK=${LORA_RANK:-0} TRAIN_LR=${TRAIN_LR:-1e-5}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}_route_a_qwen35_9b}
exec bash "$PROJECT_ROOT/scripts/smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh"
