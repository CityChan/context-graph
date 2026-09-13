#!/bin/bash

# One real FoldGRPO optimizer step inside an existing four-node idev:
# one search node plus three trainer ranks.

set -euo pipefail

export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
export RUN_TAG=${RUN_TAG:-idev_4n_smoke_1step_32k_active}
export PROMPT_LENGTH=${PROMPT_LENGTH:-8192}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-24576}
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-3}
export ROLLOUT_N=${ROLLOUT_N:-2}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TEST_FREQ=${TEST_FREQ:-0}
export SAVE_FREQ=${SAVE_FREQ:-0}
export TRAIN_LR=${TRAIN_LR:-1e-6}
export USE_KL_LOSS=${USE_KL_LOSS:-False}
export ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.0}
export ALGORITHM_KL_COEF=${ALGORITHM_KL_COEF:-0.0}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
export LORA_RANK=${LORA_RANK:-0}
export LORA_ALPHA=${LORA_ALPHA:-16}
export LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/train_foldagent_bc_8b_${RUN_TAG}}

bash scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh

if [ "$LORA_RANK" -gt 0 ] && [ "$SAVE_FREQ" -gt 0 ]; then
  ADAPTER_DIR="$CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS/actor/lora_adapter"
  if [ ! -d "$ADAPTER_DIR" ]; then
    echo "ERROR: FoldAgent LoRA adapter checkpoint is missing: $ADAPTER_DIR"
    exit 1
  fi
  echo "FOLDAGENT_LORA_SMOKE_OK checkpoint=$ADAPTER_DIR"
fi
