#!/bin/bash
# Pure single-turn verl-vs-TRL control. No ContextGraph agent loop, graph
# controller, graph credit, tool calls, or frozen reference policy is created.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export GSM8K_DATA_DIR=${GSM8K_DATA_DIR:-${SCRATCH:?SCRATCH must be set}/context-graph-data/gsm8k-qerl-xml}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
export ROLLOUT_N=${ROLLOUT_N:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
export PROMPT_LENGTH=${PROMPT_LENGTH:-1024}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-7473}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TRAIN_LR=${TRAIN_LR:-1e-5}
export LORA_RANK=${LORA_RANK:-32}
export LORA_ALPHA=${LORA_ALPHA:-32}
export USE_KL_LOSS=${USE_KL_LOSS:-False}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}
export OPTIMIZER=${OPTIMIZER:-AdamW8bit}
export OPTIMIZER_IMPL=${OPTIMIZER_IMPL:-bitsandbytes.optim}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
export ADAM_BETA1=${ADAM_BETA1:-0.9}
export ADAM_BETA2=${ADAM_BETA2:-0.99}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
export CLIP_GRAD=${CLIP_GRAD:-0.2}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
export DATA_PROMPT_STYLE=${DATA_PROMPT_STYLE:-qerl_xml}
export CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-$SCRIPT_DIR/gsm8k_qerl_xml_reward.py}
export TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
export RUN_TAG=${RUN_TAG:-gsm8k-qwen25-1p5b-verl-grpo-qerlmatched-lora32-200step}

exec bash "$SCRIPT_DIR/smoke_train_gsm8k_grpo_1node_10step.sh"
