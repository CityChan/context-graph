#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AGENT_KIND=contextgraph
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export SAVE_STEPS=${SAVE_STEPS:-50}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-0}
exec bash "$SCRIPT_DIR/train_gsm8k_trl_agent_lora32.sh"
