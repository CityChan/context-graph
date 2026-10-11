#!/bin/bash
# Local DeepSeek teacher, explicit policy decisions, official ScienceWorld train.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run inside an existing four-node idev allocation}"
export DOMAIN=scienceworld EXPLICIT_GRAPH_SFT=1
export EXPECTED_NUM_NODES=4 TEACHER_TP=4
export STUDENT_TOKENIZER_ID=${STUDENT_TOKENIZER_ID:-Qwen/Qwen3.5-9B}
export SAMPLES=${SAMPLES:-3} CANDIDATES=${CANDIDATES:-2}
export NUM_WORKERS=${WORKERS:-1} MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export PROMPT_LENGTH=${PROMPT_LENGTH:-8192} RESPONSE_LENGTH=${RESPONSE_LENGTH:-24576}
export MAX_TURN=${MAX_TURN:-100} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID}-scienceworld-explicit-$(date +%Y%m%d_%H%M%S)}
exec bash "$SCRIPT_DIR/generate_ctxgraph_sft_deepseek_v4_interactive_8node.sh"
