#!/bin/bash
# Real evaluation controller + automatic history management; successful train episodes only.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run inside an idle existing four-node idev allocation}"
BENCHMARK=${1:-scienceworld}
case "$BENCHMARK" in scienceworld|bcp) ;; *) echo "Usage: $0 [scienceworld|bcp]" >&2; exit 2 ;; esac
unset RUN_DIR ARTIFACT_ROOT RUN_TAG
export RUN_TAG="${SLURM_JOB_ID}-${BENCHMARK}-auto-$(date +%Y%m%d_%H%M%S)"
export SFT_GRAPH_PROTOCOL=controller SFT_TEACHER_PROFILE=evaluation AGENT_BUDGET_MODE=working_context
export SFT_CURATION_SMOKE=1
export SAMPLES=${SAMPLES:-3} CANDIDATES=${CANDIDATES:-2} WORKERS=1
export EXPECTED_NUM_NODES=4 TEACHER_TP=4 MAX_NUM_SEQS=1 NUM_WORKERS=1
export STUDENT_TOKENIZER_ID=${STUDENT_TOKENIZER_ID:-Qwen/Qwen3.5-9B}
export PROMPT_LENGTH=${PROMPT_LENGTH:-8192} RESPONSE_LENGTH=${RESPONSE_LENGTH:-24576}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((PROMPT_LENGTH + RESPONSE_LENGTH))}
export MAX_TURN=${MAX_TURN:-100} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
echo "AUTO_GRAPH_SFT_SMOKE benchmark=$BENCHMARK questions=$SAMPLES candidates=$CANDIDATES protocol=controller automatic_history=enabled"
if [[ "$BENCHMARK" == scienceworld ]]; then
    export SFT_SELECTION=task_balanced
    exec bash "$SCRIPT_DIR/generate_scienceworld_graph_sft_deepseek_4node_idev.sh"
fi
# This legacy selector reuses the durable collector/server; SFT_GRAPH_PROTOCOL
# determines the actual runtime protocol and curator, not this launcher switch.
export BCP_EXPLICIT_GRAPH_SFT=1 COLOCATE_SEARCH=1
exec bash "$SCRIPT_DIR/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh"
