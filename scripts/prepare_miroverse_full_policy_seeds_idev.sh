#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
INPUT=${INPUT:-$SCRATCH/datasets/MiroVerse-v0.1/jsonl_sft/MiroVerse-MuSiQue.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/contextgraph_sft/miroverse_full_policy}
MAX_SAMPLES=${MAX_SAMPLES:-0}
CORPUS_OUTPUT=${CORPUS_OUTPUT:-$OUTPUT_DIR/retrieval_corpus.parquet}
CORPUS_MANIFEST=${CORPUS_MANIFEST:-$OUTPUT_DIR/retrieval_corpus_manifest.json}
EMBEDDINGS_OUTPUT=${EMBEDDINGS_OUTPUT:-$OUTPUT_DIR/retrieval_embeddings.pkl}
EMBEDDINGS_MANIFEST=${EMBEDDINGS_MANIFEST:-$OUTPUT_DIR/retrieval_embeddings_manifest.json}
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}
EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-16}
EMBED_MAX_LENGTH=${EMBED_MAX_LENGTH:-2048}
EMBED_ATTN_IMPLEMENTATION=${EMBED_ATTN_IMPLEMENTATION:-sdpa}
BUILD_EMBEDDINGS=${BUILD_EMBEDDINGS:-1}
SEARCH_HF_HOME=${SEARCH_HF_HOME:-/work/09281/chc_1996/vista/cache}
SEARCH_HF_HUB_CACHE=${SEARCH_HF_HUB_CACHE:-$SEARCH_HF_HOME/hub}

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
python -u scripts/prepare_miroverse_retrieval_corpus.py --input "$INPUT" --output "$CORPUS_OUTPUT" --manifest "$CORPUS_MANIFEST" --max-samples "$MAX_SAMPLES"
if [ "$BUILD_EMBEDDINGS" = "1" ]; then
  HF_HOME="$SEARCH_HF_HOME" HF_HUB_CACHE="$SEARCH_HF_HUB_CACHE" python -u scripts/embed_local_search_corpus.py --input "$CORPUS_OUTPUT" --output "$EMBEDDINGS_OUTPUT" --manifest "$EMBEDDINGS_MANIFEST" --model "$EMBED_MODEL" --batch-size "$EMBED_BATCH_SIZE" --max-length "$EMBED_MAX_LENGTH" --attn-implementation "$EMBED_ATTN_IMPLEMENTATION"
fi
python -u scripts/prepare_miroverse_search_policy_seeds.py --input "$INPUT" --output "$OUTPUT_DIR/seeds.parquet" --manifest "$OUTPUT_DIR/seeds_manifest.json" --max-samples "$MAX_SAMPLES"
