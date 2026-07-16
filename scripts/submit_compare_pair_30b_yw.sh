#!/bin/bash
set -euo pipefail

TASK=${1:-}
if [ "$TASK" != "alfworld" ]; then
  echo "Usage: $0 <alfworld> [steps] [walltime]"
  exit 2
fi

TOTAL_TRAINING_STEPS=${2:-}
SBATCH_TIME=${3:-12:00:00}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
if [[ "${SCRATCH:-}" == */vista ]]; then
  DEFAULT_SCRATCH_BASE=$SCRATCH
elif [ -n "${SCRATCH:-}" ]; then
  DEFAULT_SCRATCH_BASE=${SCRATCH}/vista
else
  DEFAULT_SCRATCH_BASE=/scratch/07144/${USER:-$(whoami)}/vista
fi
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
BASE=${CHECKPOINT_BASE:-${DEFAULT_SCRATCH_BASE}/checkpoints/context-graph-compare}
NS=${HF_NAMESPACE:-lingchensanwen}

cd "$PROJECT_ROOT"

stamp=$(date +%Y%m%d_%H%M%S)
common_exports=(ALL TASK="$TASK")
if [ -n "$TOTAL_TRAINING_STEPS" ]; then
  common_exports+=(TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS")
fi

submit_one() {
  local agent=$1
  local jobname=$2
  local exp="${agent}_${TASK}_30b_8n_compare_${stamp}"
  local export_arg

  export_arg=$(IFS=,; echo "${common_exports[*]},AGENT=${agent},EXPERIMENT_NAME=${exp}")
  local submit_out
  submit_out=$(sbatch \
    --job-name="$jobname" \
    --output="${jobname}.%j.out" \
    --error="${jobname}.%j.err" \
    --time="$SBATCH_TIME" \
    --export="$export_arg" \
    scripts/run_compare_30b_8n_yw.sh)
  local job
  job=$(echo "$submit_out" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$job" ]; then
    echo "Could not parse job id from: $submit_out" >&2
    exit 3
  fi

  local root="${BASE}/${exp}"
  local repo="${NS}/${exp}"
  nohup env UPLOAD_ON_TIMEOUT=1 bash scripts/wait_and_upload_smoke_checkpoint.sh "$job" "$root" "$repo" "$exp" \
    > "upload_watch_${job}.log" 2>&1 &
  local watcher=$!

  echo "${agent}: job=${job} watcher=${watcher} exp=${exp} root=${root} repo=${repo}"
}

submit_one ctxgraph "cg-${TASK}-30b"
submit_one baseline "base-${TASK}-30b"

squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.32j %.2t %.10M %.10L %.6D %R"
