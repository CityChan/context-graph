#!/bin/bash
# Submit the three full ScienceAgentBench Qwen3-8B evaluations.
# Run this wrapper from a TACC login node, not from an idev/compute node.

set -euo pipefail

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit_eval_sab_8b_4node_formal.sh must run on a login node, not inside a Slurm allocation."
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is unavailable. Log in to a TACC login node and retry."
  exit 1
fi

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
JOB_TIME=${JOB_TIME:-04:00:00}
SAB_DATA_SEED=${SAB_DATA_SEED:-42}
SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:--1}
SAB_CTXGRAPH_PROTOCOL=${SAB_CTXGRAPH_PROTOCOL:-legacy}
EVAL_SCRIPT=scripts/eval_sab_react_8b_4node_smoke.sh

cd "$PROJECT_ROOT"
mkdir -p logs

if [ ! -f "$EVAL_SCRIPT" ]; then
  echo "ERROR: missing $PROJECT_ROOT/$EVAL_SCRIPT"
  exit 1
fi

echo "Submitting full SAB Qwen3-8B evaluations:"
echo "  project=$PROJECT_ROOT"
echo "  methods=react,fold,ctxgraph"
echo "  nodes=4 per job"
echo "  time=$JOB_TIME per job"
echo "  val_max_samples=$SAB_VAL_MAX_SAMPLES"
echo "  data_seed=$SAB_DATA_SEED"
echo "  real_eval=1"
echo "  ctxgraph_protocol=$SAB_CTXGRAPH_PROTOCOL"

for method in react fold ctxgraph; do
  method_protocol=legacy
  if [ "$method" = "ctxgraph" ]; then method_protocol=$SAB_CTXGRAPH_PROTOCOL; fi
  job_id=$(sbatch --parsable --job-name="eval-sab-${method}-${method_protocol}-8b-formal" --output="logs/eval-sab-${method}-${method_protocol}-8b-formal.%j.out" --error="logs/eval-sab-${method}-${method_protocol}-8b-formal.%j.err" --nodes=4 --time="$JOB_TIME" --export=ALL,SAB_METHOD="$method",SAB_CTXGRAPH_PROTOCOL="$method_protocol",SAB_RUN_TAG=formal,SAB_REAL_EVAL=1,SAB_VAL_MAX_SAMPLES="$SAB_VAL_MAX_SAMPLES",SAB_DATA_SEED="$SAB_DATA_SEED" "$EVAL_SCRIPT")
  printf '%s: submitted job %s\n' "$method" "$job_id"
done

echo "All three jobs submitted. Check them with: squeue -u $USER"
