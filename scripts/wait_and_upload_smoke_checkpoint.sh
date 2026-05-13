#!/bin/bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <job_id> <checkpoint_root> <hf_repo_id> <experiment_name>"
  exit 2
fi

JOB_ID=$1
CHECKPOINT_ROOT=$2
HF_REPO_ID=$3
EXPERIMENT_NAME=$4

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEFAULT_WORK_BASE=${WORK:-/work/07144/${USER:-$(whoami)}/vista}
if [[ "${SCRATCH:-}" == */vista ]]; then
  DEFAULT_SCRATCH_BASE=$SCRATCH
elif [ -n "${SCRATCH:-}" ]; then
  DEFAULT_SCRATCH_BASE=${SCRATCH}/vista
else
  DEFAULT_SCRATCH_BASE=/scratch/07144/${USER:-$(whoami)}/vista
fi
PROJECT_ROOT=${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}
CONDA_ROOT=${CONDA_ROOT:-${DEFAULT_WORK_BASE}/miniconda3}
HF_HOME=${HF_HOME:-${DEFAULT_WORK_BASE}/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-${DEFAULT_SCRATCH_BASE}/cache}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-86400}
POLL_SECONDS=${POLL_SECONDS:-60}
UPLOAD_ON_TIMEOUT=${UPLOAD_ON_TIMEOUT:-0}

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate cxtgraph
cd "$PROJECT_ROOT"
export HF_HOME XDG_CACHE_HOME

echo "Watching job $JOB_ID"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "HF repo: $HF_REPO_ID"
echo "Experiment: $EXPERIMENT_NAME"
echo "Started watcher: $(date)"

deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
state=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  state=$(sacct -j "$JOB_ID" -X -n -o State 2>/dev/null | awk 'NF {print $1; exit}' || true)
  if [ -z "$state" ]; then
    if squeue -h -j "$JOB_ID" >/dev/null 2>&1; then
      state="PENDING/RUNNING"
    fi
  fi

  case "$state" in
    COMPLETED)
      echo "Job $JOB_ID completed at $(date)"
      break
      ;;
    TIMEOUT)
      if [ "$UPLOAD_ON_TIMEOUT" = "1" ]; then
        echo "Job $JOB_ID timed out; will try uploading latest completed checkpoint."
        break
      fi
      echo "Job $JOB_ID ended with state=$state; not uploading."
      exit 10
      ;;
    FAILED|CANCELLED|CANCELLED+|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
      echo "Job $JOB_ID ended with state=$state; not uploading."
      exit 10
      ;;
    *)
      echo "$(date): job $JOB_ID state=${state:-unknown}; waiting..."
      sleep "$POLL_SECONDS"
      ;;
  esac
done

if [ "$state" != "COMPLETED" ] && { [ "$state" != "TIMEOUT" ] || [ "$UPLOAD_ON_TIMEOUT" != "1" ]; }; then
  echo "Timed out waiting for job $JOB_ID; last state=${state:-unknown}"
  exit 11
fi

if [ ! -d "$CHECKPOINT_ROOT" ]; then
  echo "Checkpoint root missing: $CHECKPOINT_ROOT"
  exit 12
fi

echo "Uploading latest checkpoint at $(date)"
bash scripts/sync_latest_checkpoint_to_hf.sh "$CHECKPOINT_ROOT" "$HF_REPO_ID" "$EXPERIMENT_NAME" false
echo "Upload watcher finished at $(date)"
