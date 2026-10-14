#!/bin/bash
# Full BC-P train collection; local DeepSeek teacher and colocated retrieval.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run inside an idle four-node idev allocation}"
: "${SCRATCH:?Missing SCRATCH}"
export PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$PROJECT_ROOT"
commit=$(git rev-parse HEAD)
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
export PROMPT_LENGTH=${PROMPT_LENGTH:-8192}
[[ "$CONTEXT_LENGTH" =~ ^[1-9][0-9]*$ && "$PROMPT_LENGTH" =~ ^[1-9][0-9]*$ ]] || exit 2
(( CONTEXT_LENGTH > PROMPT_LENGTH )) || { echo 'Context must exceed prompt allocation' >&2; exit 2; }
export RESPONSE_LENGTH=$((CONTEXT_LENGTH - PROMPT_LENGTH)) MAX_MODEL_LEN=$CONTEXT_LENGTH
export AUTO_SFT_RUN_DIR=${AUTO_SFT_RUN_DIR:-$SCRATCH/contextgraph_sft/bcp-auto-full/$commit/context-$CONTEXT_LENGTH}
export RUN_DIR=$AUTO_SFT_RUN_DIR ARTIFACT_ROOT=$AUTO_SFT_RUN_DIR
export RUN_TAG="bcp-auto-full-${commit:0:12}-$CONTEXT_LENGTH"
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/collection.lock"
flock -n 9 || { echo "Another collector holds $RUN_DIR/collection.lock" >&2; exit 2; }
export SFT_GRAPH_PROTOCOL=controller SFT_TEACHER_PROFILE=evaluation
export AGENT_BUDGET_MODE=working_context SFT_CURATION_SMOKE=0 SFT_REQUIRE_COMPLETE=1
export BCP_EXPLICIT_GRAPH_SFT=1 COLOCATE_SEARCH=1
export SAMPLES=-1 START_INDEX=0 CANDIDATES=${CANDIDATES:-2} WORKERS=1
export EXPECTED_NUM_NODES=4 TEACHER_TP=4 NUM_WORKERS=1 MAX_NUM_SEQS=1
export STUDENT_TOKENIZER_ID=${STUDENT_TOKENIZER_ID:-Qwen/Qwen3.5-9B}
export DATA_PATH=${DATA_PATH:-data/bc_train.parquet} HELDOUT_PATH=${HELDOUT_PATH:-data/bc_test.parquet}
export MAX_TURN=${MAX_TURN:-100} GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
unset ARRAY_CHUNK_MODE DRY_RUN TEACHER_PREFLIGHT_ONLY PREFLIGHT_ONLY
echo "BCP_AUTO_GRAPH_SFT_FULL samples=all candidates=$CANDIDATES context=$CONTEXT_LENGTH commit=$commit run=$RUN_DIR"
echo "Resume with the same commit, context and AUTO_SFT_RUN_DIR=$RUN_DIR; completed episodes are skipped."
exec bash "$SCRIPT_DIR/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh"
