#!/bin/bash
set -euo pipefail

TASK_ARG=${1:-both}
SBATCH_TIME=${2:-02:00:00}
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
BASE=${CHECKPOINT_BASE:-${DEFAULT_SCRATCH_BASE}/checkpoints/context-graph-master-smoke}
NS=${HF_NAMESPACE:-lingchensanwen}
ROLLOUT_N=${ROLLOUT_N:-8}

case "$TASK_ARG" in
  alfworld) TASKS=(alfworld) ;;
  hotpotqa) TASKS=(hotpotqa) ;;
  both) TASKS=(alfworld hotpotqa) ;;
  *)
    echo "Usage: $0 [alfworld|hotpotqa|both] [walltime]"
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
stamp=$(date +%Y%m%d_%H%M%S)

submit_one() {
  local task=$1
  local jobname="cg-${task}-smoke"
  local exp="cg_${task}_30b_8n_master_smoke_ckpt_${stamp}"
  local export_arg

  export_arg="ALL,TASK=${task},EXPERIMENT_NAME=${exp},SAVE_FREQ=1,HF_SYNC_CHECKPOINTS=0,ROLLOUT_N=${ROLLOUT_N}"

  local submit_out
  submit_out=$(sbatch \
    --job-name="$jobname" \
    --output="${jobname}.%j.out" \
    --error="${jobname}.%j.err" \
    --time="$SBATCH_TIME" \
    --export="$export_arg" \
    scripts/smoke_ctxgraph_30b_8n_master_yw.sh)
  local job
  job=$(echo "$submit_out" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$job" ]; then
    echo "Could not parse job id from: $submit_out" >&2
    exit 3
  fi

  local root="${BASE}/${exp}"
  local repo="${NS}/${exp}"
  nohup bash scripts/wait_and_upload_smoke_checkpoint.sh "$job" "$root" "$repo" "$exp" \
    > "upload_watch_${job}.log" 2>&1 &
  local watcher=$!

  echo "${task}: job=${job} watcher=${watcher} exp=${exp} root=${root} repo=${repo} rollout_n=${ROLLOUT_N}"
}

for task in "${TASKS[@]}"; do
  submit_one "$task"
done

squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.32j %.2t %.10M %.10L %.6D %R"
