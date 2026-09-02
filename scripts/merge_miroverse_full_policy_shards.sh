#!/bin/bash
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SHARD_ROOT=${SHARD_ROOT:-$SCRATCH/contextgraph_sft/miroverse_full_policy_native/shards}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/contextgraph_sft/miroverse_full_policy_native/full_merged}
PLAN=${PLAN:-$SHARD_ROOT/submission_plan.tsv}

test -s "$PLAN"
COMPLETE_DIRS=()
while IFS=$'\t' read -r start count run_tag job_id dependency; do
  if [ "$start" = "start_index" ]; then continue; fi
  shard_dir="$SHARD_ROOT/$run_tag"
  if [ ! -f "$shard_dir/.complete" ]; then
    echo "ERROR: planned shard is incomplete: $run_tag (job $job_id)"
    exit 2
  fi
  COMPLETE_DIRS+=("$shard_dir")
done < "$PLAN"
if [ "${#COMPLETE_DIRS[@]}" -eq 0 ]; then echo "ERROR: submission plan contains no shards"; exit 2; fi
INPUTS=()
for shard_dir in "${COMPLETE_DIRS[@]}"; do
  mapfile -t shard_results < <(find "$shard_dir/raw" -maxdepth 1 -name 'gaia_results_*.json' -type f | sort)
  if [ "${#shard_results[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one raw result in $shard_dir, found ${#shard_results[@]}"
    exit 2
  fi
  INPUTS+=("${shard_results[0]}")
done

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
python -u scripts/build_contextgraph_full_policy_sft.py "${INPUTS[@]}" --output "$OUTPUT_DIR/contextgraph_sft_train.parquet" --validation-output "$OUTPUT_DIR/contextgraph_sft_validation.parquet" --manifest "$OUTPUT_DIR/manifest.json"
echo "Merged ${#INPUTS[@]} completed shards into $OUTPUT_DIR"
