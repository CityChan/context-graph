#!/bin/bash
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SEEDS_MANIFEST=${SEEDS_MANIFEST:-$SCRATCH/contextgraph_sft/miroverse_full_policy/seeds_manifest.json}
SHARD_SIZE=${SHARD_SIZE:-1500}
START_INDEX=${START_INDEX:-0}
SHARD_ROOT=${SHARD_ROOT:-$SCRATCH/contextgraph_sft/miroverse_full_policy_native/shards}

test -s "$SEEDS_MANIFEST"
if [ "$SHARD_SIZE" -le 0 ] || [ "$START_INDEX" -lt 0 ]; then
  echo "ERROR: SHARD_SIZE must be positive and START_INDEX non-negative"
  exit 2
fi

TOTAL_SAMPLES=$(python -c "import json; print(int(json.load(open('$SEEDS_MANIFEST'))['accepted']))")
if [ "$START_INDEX" -ge "$TOTAL_SAMPLES" ]; then
  echo "ERROR: START_INDEX=$START_INDEX is outside TOTAL_SAMPLES=$TOTAL_SAMPLES"
  exit 2
fi

mkdir -p "$SHARD_ROOT"
PLAN="$SHARD_ROOT/submission_plan.tsv"
printf 'start_index\tmax_samples\trun_tag\tjob_id\tdependency\n' > "$PLAN"
previous_job=""
start=$START_INDEX
while [ "$start" -lt "$TOTAL_SAMPLES" ]; do
  remaining=$((TOTAL_SAMPLES - start))
  count=$SHARD_SIZE
  if [ "$remaining" -lt "$count" ]; then count=$remaining; fi
  run_tag="miroverse_full_policy_s${start}_n${count}"
  artifact_root="$SHARD_ROOT/$run_tag"
  dependency=()
  dependency_label=""
  if [ -n "$previous_job" ]; then
    dependency=(--dependency="afterok:$previous_job")
    dependency_label="afterok:$previous_job"
  fi
  job_id=$(sbatch --parsable "${dependency[@]}" --export="ALL,START_INDEX=$start,MAX_SAMPLES=$count,RUN_TAG=$run_tag,ARTIFACT_ROOT=$artifact_root" "$PROJECT_ROOT/scripts/generate_miroverse_full_policy_deepseek_v4_4node_batch.sh")
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\t%s\n' "$start" "$count" "$run_tag" "$job_id" "$dependency_label" | tee -a "$PLAN"
  previous_job=$job_id
  start=$((start + count))
done
echo "Submitted serial four-node shards; plan: $PLAN"
