#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
DATA_DIR=${DATA_DIR:-$SCRATCH/contextgraph_sft/miroverse_remaining_full_policy}
SHARD_ROOT=${SHARD_ROOT:-$SCRATCH/contextgraph_sft/miroverse_remaining_full_policy_native/shards}
MAX_ARRAY_INDEX=${MAX_ARRAY_INDEX:-79}
MIROVERSE_ROOT=${MIROVERSE_ROOT:-$SCRATCH/datasets/MiroVerse-v0.1/jsonl_sft}

remaining_files=$(find "$MIROVERSE_ROOT" -maxdepth 1 -type f -name 'MiroVerse-*.jsonl' ! -name 'MiroVerse-MuSiQue.jsonl' | wc -l)
if [ "$remaining_files" -lt 11 ]; then
  echo "ERROR: expected 11 remaining MiroVerse JSONL files under $MIROVERSE_ROOT, found $remaining_files"
  echo "Download them before submitting the preparation and generation jobs."
  exit 2
fi

cd "$PROJECT_ROOT"
prep_job=$(sbatch --parsable --export="ALL,OUTPUT_DIR=$DATA_DIR,MIROVERSE_ROOT=$MIROVERSE_ROOT" scripts/prepare_miroverse_remaining_full_policy_1node_batch.sh)
generation_job=$(sbatch --parsable --dependency="afterok:$prep_job" --array="0-$MAX_ARRAY_INDEX%1" --export="ALL,SEEDS_MANIFEST=$DATA_DIR/seeds_manifest.json,DATA_PATH=$DATA_DIR/seeds.parquet,LOCAL_SEARCH_CORPUS=$DATA_DIR/retrieval_corpus.parquet,LOCAL_SEARCH_EMBEDDINGS=$DATA_DIR/retrieval_embeddings.pkl,SHARD_ROOT=$SHARD_ROOT" scripts/generate_miroverse_full_policy_deepseek_v4_5node_batch.sh)
printf 'prep_job=%s\ngeneration_array=%s\ndata_dir=%s\nshard_root=%s\n' "$prep_job" "$generation_job" "$DATA_DIR" "$SHARD_ROOT"
