#!/bin/bash
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SHARD_ROOT=${SHARD_ROOT:-$SCRATCH/contextgraph_sft/miroverse_full_policy_native/shards}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/contextgraph_sft/miroverse_full_policy_native/full_merged}
SEEDS_MANIFEST=${SEEDS_MANIFEST:-$SCRATCH/contextgraph_sft/miroverse_full_policy/seeds_manifest.json}

test -s "$SEEDS_MANIFEST"
TOTAL_SAMPLES=$(python -c "import json; print(int(json.load(open('$SEEDS_MANIFEST'))['accepted']))")
mapfile -t COMPLETE_DIRS < <(find "$SHARD_ROOT" -mindepth 2 -maxdepth 2 -name .complete -type f -printf '%h\n' | sort -V)
if [ "${#COMPLETE_DIRS[@]}" -eq 0 ]; then echo "ERROR: no completed shards found under $SHARD_ROOT"; exit 2; fi
INPUTS=()
expected_start=0
for shard_dir in "${COMPLETE_DIRS[@]}"; do
  request="$shard_dir/shard_request.txt"
  test -s "$request"
  start=$(awk -F= '$1 == "start_index" {print $2}' "$request")
  count=$(awk -F= '$1 == "max_samples" {print $2}' "$request")
  if [ "$start" -ne "$expected_start" ]; then
    echo "ERROR: shard coverage gap or overlap: expected start $expected_start, found $start in $shard_dir"
    exit 2
  fi
  expected_start=$((expected_start + count))
  mapfile -t shard_results < <(find "$shard_dir/raw" -maxdepth 1 -name 'gaia_results_*.json' -type f | sort)
  if [ "${#shard_results[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one raw result in $shard_dir, found ${#shard_results[@]}"
    exit 2
  fi
  INPUTS+=("${shard_results[0]}")
done
if [ "$expected_start" -ne "$TOTAL_SAMPLES" ]; then
  echo "ERROR: completed shard coverage ends at $expected_start, expected $TOTAL_SAMPLES"
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
python -u scripts/build_contextgraph_full_policy_sft.py "${INPUTS[@]}" --output "$OUTPUT_DIR/contextgraph_sft_train.parquet" --validation-output "$OUTPUT_DIR/contextgraph_sft_validation.parquet" --manifest "$OUTPUT_DIR/manifest.json"
echo "Merged ${#INPUTS[@]} completed shards into $OUTPUT_DIR"
