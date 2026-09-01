#!/bin/bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
INPUT=${INPUT:-$SCRATCH/datasets/MiroVerse-v0.1/jsonl_sft/MiroVerse-MuSiQue.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/contextgraph_sft/miroverse_full_policy}
MAX_SAMPLES=${MAX_SAMPLES:-0}

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
python -u scripts/prepare_miroverse_search_policy_seeds.py --input "$INPUT" --output "$OUTPUT_DIR/seeds.parquet" --manifest "$OUTPUT_DIR/seeds_manifest.json" --max-samples "$MAX_SAMPLES"
