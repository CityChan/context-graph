#!/bin/bash
# Run a conservative DeepSeek-V4-Flash-0731 ContextGraph smoke inside an
# existing four-GH200 Vista idev allocation. The underlying production script
# discovers all four allocated nodes, starts a Ray TP=4 + expert-parallel vLLM
# server, and runs one interactive domain. Run ALFWorld and ScienceWorld
# sequentially by overriding DOMAIN.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
export TEACHER_TP=${TEACHER_TP:-4}
export DOMAIN=${DOMAIN:-alfworld}
export MAX_SAMPLES=${MAX_SAMPLES:-2}
export NUM_WORKERS=${NUM_WORKERS:-1}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
export PROMPT_LENGTH=${PROMPT_LENGTH:-12288}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-16384}
export MAX_TURN=${MAX_TURN:-30}
export REASONING_EFFORT=${REASONING_EFFORT:-non-thinking}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}_deepseek_v4_4node}

exec bash "$SCRIPT_DIR/generate_ctxgraph_sft_deepseek_v4_interactive_8node.sh"
