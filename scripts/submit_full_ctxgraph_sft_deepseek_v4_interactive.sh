#!/bin/bash
# Submit the complete ALFWorld and ScienceWorld interactive trajectory sets as
# restart-safe shards. Each shard writes its own raw JSON and filtered parquet.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
GENERATOR_SCRIPT=${GENERATOR_SCRIPT:-$PROJECT_ROOT/scripts/generate_ctxgraph_sft_deepseek_v4_interactive_8node.sh}
CHUNK_SIZE=${CHUNK_SIZE:-1000}
MAX_CONCURRENT_PER_DOMAIN=${MAX_CONCURRENT_PER_DOMAIN:-1}
ALFWORLD_TOTAL=${ALFWORLD_TOTAL:-3553}
# ScienceWorld has 7,207 variations across train/dev/test; the official train
# split is approximately half. Only the resulting array end matters here, and
# the final shard naturally stops at the installed train split's actual size.
SCIENCEWORLD_TRAIN_TOTAL=${SCIENCEWORLD_TRAIN_TOTAL:-3604}
DRY_RUN=${DRY_RUN:-0}

test -x "$(command -v sbatch)" || { echo "ERROR: sbatch is unavailable; run this on a Vista login node"; exit 2; }
test -f "$GENERATOR_SCRIPT" || { echo "ERROR: generator script not found: $GENERATOR_SCRIPT"; exit 2; }
test "$CHUNK_SIZE" -gt 0 || { echo "ERROR: CHUNK_SIZE must be positive"; exit 2; }
test "$MAX_CONCURRENT_PER_DOMAIN" -gt 0 || { echo "ERROR: MAX_CONCURRENT_PER_DOMAIN must be positive"; exit 2; }
test "$ALFWORLD_TOTAL" -gt 0 || { echo "ERROR: ALFWORLD_TOTAL must be positive"; exit 2; }
test "$SCIENCEWORLD_TRAIN_TOTAL" -gt 0 || { echo "ERROR: SCIENCEWORLD_TRAIN_TOTAL must be positive"; exit 2; }

ALF_END=$(( (ALFWORLD_TOTAL - 1) / CHUNK_SIZE ))
SCI_END=$(( (SCIENCEWORLD_TRAIN_TOTAL - 1) / CHUNK_SIZE ))
ALF_ARRAY=0-${ALF_END}%${MAX_CONCURRENT_PER_DOMAIN}
SCI_ARRAY=0-${SCI_END}%${MAX_CONCURRENT_PER_DOMAIN}

# Prevent interactive-shell state from overriding per-array isolation.
unset RAY_ADDRESS DG_JIT_CACHE_DIR VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE
unset MODEL_PATH STUDENT_TOKENIZER_PATH DOMAIN ARRAY_CHUNK_MODE MAX_SAMPLES START_INDEX
unset RUN_TAG ARTIFACT_ROOT

echo "Full DeepSeek-V4 interactive ContextGraph trajectory generation"
echo "ALFWorld: total=$ALFWORLD_TOTAL chunk=$CHUNK_SIZE array=$ALF_ARRAY"
echo "ScienceWorld train planning total=$SCIENCEWORLD_TRAIN_TOTAL chunk=$CHUNK_SIZE array=$SCI_ARRAY"
echo "Maximum simultaneous generation jobs: $((2 * MAX_CONCURRENT_PER_DOMAIN))"
echo "Maximum simultaneous GH200 nodes: $((16 * MAX_CONCURRENT_PER_DOMAIN))"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN: sbatch --array=$ALF_ARRAY --export=ALL,DOMAIN=alfworld,ARRAY_CHUNK_MODE=1,MAX_SAMPLES=$CHUNK_SIZE $GENERATOR_SCRIPT"
  echo "DRY RUN: sbatch --array=$SCI_ARRAY --export=ALL,DOMAIN=scienceworld,ARRAY_CHUNK_MODE=1,MAX_SAMPLES=$CHUNK_SIZE $GENERATOR_SCRIPT"
  exit 0
fi

ALF_JOB=$(sbatch --parsable --job-name=cg-sft-dsv4-alf-full --array="$ALF_ARRAY" --export=ALL,DOMAIN=alfworld,ARRAY_CHUNK_MODE=1,MAX_SAMPLES="$CHUNK_SIZE" "$GENERATOR_SCRIPT")
SCI_JOB=$(sbatch --parsable --job-name=cg-sft-dsv4-sci-full --array="$SCI_ARRAY" --export=ALL,DOMAIN=scienceworld,ARRAY_CHUNK_MODE=1,MAX_SAMPLES="$CHUNK_SIZE" "$GENERATOR_SCRIPT")

echo "Submitted ALFWorld array: $ALF_JOB"
echo "Submitted ScienceWorld array: $SCI_JOB"
echo "Monitor with: squeue -j $ALF_JOB,$SCI_JOB"
echo "Shard outputs: $SCRATCH/contextgraph_sft/deepseek_v4_flash_0731_interactive/<array_job_id>_<shard_index>/<domain>"
