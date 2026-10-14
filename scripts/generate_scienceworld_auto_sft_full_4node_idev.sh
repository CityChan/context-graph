#!/bin/bash
# Full official train collection, stable output and counted controller decisions.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run inside an idle four-node idev allocation}"
: "${SCRATCH:?Missing SCRATCH}"
export PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$PROJECT_ROOT"
# Pin this commit when resuming; never append episodes produced by changed code.
commit=$(git rev-parse HEAD)
export AUTO_SFT_RUN_DIR=${AUTO_SFT_RUN_DIR:-$SCRATCH/contextgraph_sft/scienceworld-auto-full/$commit}
export RUN_DIR=$AUTO_SFT_RUN_DIR ARTIFACT_ROOT=$AUTO_SFT_RUN_DIR
export RUN_TAG="scienceworld-auto-full-${commit:0:12}"
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/collection.lock"
flock -n 9 || { echo "Another collector holds $RUN_DIR/collection.lock" >&2; exit 2; }
export SFT_GRAPH_PROTOCOL=controller SFT_TEACHER_PROFILE=evaluation SFT_SELECTION=task_balanced
export AGENT_BUDGET_MODE=working_context SFT_CURATION_SMOKE=0 SFT_REQUIRE_COMPLETE=1
export SAMPLES=-1 START_INDEX=0 CANDIDATES=${CANDIDATES:-2} WORKERS=1
export EXPECTED_NUM_NODES=4 TEACHER_TP=4 NUM_WORKERS=1 MAX_NUM_SEQS=1
export STUDENT_TOKENIZER_ID=${STUDENT_TOKENIZER_ID:-Qwen/Qwen3.5-9B}
export PROMPT_LENGTH=${PROMPT_LENGTH:-8192} RESPONSE_LENGTH=${RESPONSE_LENGTH:-24576}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((PROMPT_LENGTH + RESPONSE_LENGTH))}
export MAX_TURN=${MAX_TURN:-100} MAX_STEPS=${MAX_STEPS:-100}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
unset ARRAY_CHUNK_MODE DRY_RUN TEACHER_PREFLIGHT_ONLY PREFLIGHT_ONLY
echo "AUTO_GRAPH_SFT_FULL samples=all candidates=$CANDIDATES commit=$commit run=$RUN_DIR"
echo "Resume with the same commit and AUTO_SFT_RUN_DIR=$RUN_DIR; completed episodes are skipped."
exec bash "$SCRIPT_DIR/generate_scienceworld_graph_sft_deepseek_4node_idev.sh"
