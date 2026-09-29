#!/bin/bash
# Offline tokenizer regression check; no model weights, GPU, or generation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export QWEN_TOKENIZER_PATH=${QWEN_TOKENIZER_PATH:-Qwen/Qwen3-8B}
python -c 'import transformers, torch'
python -m unittest discover -s tests -p test_agent_utils_chat_template.py -v
