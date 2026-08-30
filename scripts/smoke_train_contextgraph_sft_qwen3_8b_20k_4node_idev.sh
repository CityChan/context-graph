#!/bin/bash

# Stress-test one full-parameter optimizer step on the four longest formal
# ContextGraph SFT trajectories before launching the complete 20K run.
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
FORMAL_DATA_DIR=${FORMAL_DATA_DIR:-$SCRATCH/contextgraph_sft/qwen3_8b_formal_945161_945162}

export TRAIN_FILE=${TRAIN_FILE:-$FORMAL_DATA_DIR/contextgraph_sft_longest4.parquet}
export VAL_FILE=${VAL_FILE:-$FORMAL_DATA_DIR/contextgraph_sft_validation.parquet}
export MAX_LENGTH=20480
export TRAIN_MAX_SAMPLES=4
export VAL_MAX_SAMPLES=4
export TOTAL_TRAINING_STEPS=1
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}_qwen3_8b_sft_20k_longest_smoke}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH/contextgraph_sft_checkpoints/$RUN_TAG}

exec bash "$PROJECT_ROOT/scripts/smoke_train_contextgraph_sft_qwen3_8b_4node_idev.sh"
