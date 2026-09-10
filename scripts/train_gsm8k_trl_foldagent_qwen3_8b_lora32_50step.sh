#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AGENT_KIND=foldagent
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
export SAVE_STEPS=${SAVE_STEPS:-10}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-128}
export RUN_TAG=${RUN_TAG:-trl-foldagent-gsm8k-qwen3-8b-lora32-50step}
export WANDB_TAGS=${WANDB_TAGS:-trl,agent,foldagent,gsm8k,qwen3-8b,lora32,g16,50step}

exec bash "$SCRIPT_DIR/train_gsm8k_trl_agent_lora32.sh"
