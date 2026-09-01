#!/bin/bash
# Run the complete-policy MiroVerse smoke inside an existing four-node idev
# allocation by colocating retrieval with the first DeepSeek-V4 rank.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
export TEACHER_TP=${TEACHER_TP:-4}
export COLOCATE_SEARCH=1
export DATA_PATH=${DATA_PATH:-$SCRATCH/contextgraph_sft/miroverse_full_policy/seeds.parquet}
export LOCAL_SEARCH_CORPUS=${LOCAL_SEARCH_CORPUS:-$SCRATCH/contextgraph_sft/miroverse_full_policy/retrieval_corpus.parquet}
export LOCAL_SEARCH_EMBEDDINGS=${LOCAL_SEARCH_EMBEDDINGS:-$SCRATCH/contextgraph_sft/miroverse_full_policy/retrieval_embeddings.pkl}
export MAX_SAMPLES=${MAX_SAMPLES:-10}
export START_INDEX=${START_INDEX:-0}
export NUM_WORKERS=${NUM_WORKERS:-1}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
export FULL_POLICY_CURATOR=1
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}_miroverse_full_policy_4node_smoke}
export ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/miroverse_full_policy_native/$RUN_TAG}

test -s "$DATA_PATH"
test -s "$LOCAL_SEARCH_CORPUS"
test -s "$LOCAL_SEARCH_EMBEDDINGS"
exec bash "$SCRIPT_DIR/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh"
