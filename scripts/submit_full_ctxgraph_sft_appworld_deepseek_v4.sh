#!/bin/bash
# Submit official AppWorld train tasks as restart-safe DeepSeek-V4 SFT shards.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
GENERATOR_SCRIPT=${GENERATOR_SCRIPT:-$PROJECT_ROOT/scripts/generate_ctxgraph_sft_deepseek_v4_interactive_8node.sh}
APPWORLD_CONDA_ENV=${APPWORLD_CONDA_ENV:-appworld_cxtgraph}
APPWORLD_ROOT=${APPWORLD_ROOT:-${SCRATCH:?SCRATCH must be set}/contextgraph_deps/appworld}
CHUNK_SIZE=${CHUNK_SIZE:-50}
MAX_CONCURRENT=${MAX_CONCURRENT:-1}
DRY_RUN=${DRY_RUN:-0}

test -x "$(command -v sbatch)" || { echo "ERROR: sbatch is unavailable; run this on a Vista login node"; exit 2; }
test -f "$GENERATOR_SCRIPT" || { echo "ERROR: generator script not found: $GENERATOR_SCRIPT"; exit 2; }
test "$CHUNK_SIZE" -gt 0 || { echo "ERROR: CHUNK_SIZE must be positive"; exit 2; }
test "$MAX_CONCURRENT" -gt 0 || { echo "ERROR: MAX_CONCURRENT must be positive"; exit 2; }

if [ -n "${APPWORLD_TRAIN_TOTAL:-}" ]; then
  TOTAL=$APPWORLD_TRAIN_TOTAL
else
  set +u
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate "$APPWORLD_CONDA_ENV"
  set -u
  export APPWORLD_ROOT
  TOTAL=$(python "$PROJECT_ROOT/scripts/make_appworld_data.py" --count-only --seed 42)
fi
test "$TOTAL" -gt 0 || { echo "ERROR: AppWorld train split is empty"; exit 2; }

ARRAY_END=$(( (TOTAL - 1) / CHUNK_SIZE ))
ARRAY_SPEC=0-${ARRAY_END}%${MAX_CONCURRENT}
unset RAY_ADDRESS DG_JIT_CACHE_DIR VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE
unset MODEL_PATH STUDENT_TOKENIZER_PATH DOMAIN ARRAY_CHUNK_MODE MAX_SAMPLES START_INDEX
unset RUN_TAG ARTIFACT_ROOT

echo "Full AppWorld DeepSeek-V4 ContextGraph trajectory generation"
echo "AppWorld train: total=$TOTAL chunk=$CHUNK_SIZE array=$ARRAY_SPEC"
echo "Maximum simultaneous GH200 nodes: $((8 * MAX_CONCURRENT))"

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY RUN: sbatch --array=$ARRAY_SPEC --export=ALL,DOMAIN=appworld,ARRAY_CHUNK_MODE=1,MAX_SAMPLES=$CHUNK_SIZE,APPWORLD_NUM_WORKERS=1 $GENERATOR_SCRIPT"
  exit 0
fi

JOB_ID=$(sbatch --parsable --job-name=cg-sft-dsv4-app-full --array="$ARRAY_SPEC" --export=ALL,DOMAIN=appworld,ARRAY_CHUNK_MODE=1,MAX_SAMPLES="$CHUNK_SIZE",APPWORLD_NUM_WORKERS=1 "$GENERATOR_SCRIPT")
echo "Submitted AppWorld array: $JOB_ID"
echo "Monitor with: squeue -r -j $JOB_ID"
echo "Shard outputs: $SCRATCH/contextgraph_sft/deepseek_v4_flash_0731_interactive/<array_job_id>_<shard_index>/appworld"
