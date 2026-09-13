#!/bin/bash
# Full 200-step ContextGraph agent-pipeline diagnostic. This deliberately uses
# the multi-turn graph controller and is not a pure single-turn VERL framework
# control; use train_gsm8k_verl_grpo_qerlmatched_lora32_200step.sh for that.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
export ROLLOUT_N=${ROLLOUT_N:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export RUN_TAG=${RUN_TAG:-gsm8k-qwen25-1p5b-ctxgraph-graphrpo-nocredit-lora32-200step}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-gsm8k-framework-comparison}
export WANDB_TAGS=${WANDB_TAGS:-ctxgraph,verl,graphrpo,gsm8k,qwen2.5-1.5b,bf16,lora32,no-edit-credit,g16,qerl-matched}

exec bash "$SCRIPT_DIR/smoke_train_gsm8k_ctxgraph_graphrpo_nocredit_lora32_10step.sh"
