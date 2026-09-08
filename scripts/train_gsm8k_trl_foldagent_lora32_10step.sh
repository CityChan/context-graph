#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AGENT_KIND=foldagent
exec bash "$SCRIPT_DIR/train_gsm8k_trl_agent_lora32.sh"
