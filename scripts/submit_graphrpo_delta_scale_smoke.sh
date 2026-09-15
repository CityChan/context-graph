#!/bin/bash

# Calibrate frozen-reference GraphRPO credit from an existing rollout, then
# submit one fresh four-node, one-step smoke using that non-default scale.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/submit_graphrpo_delta_scale_smoke.sh BASELINE_ROLLOUT_DIR"
  exit 2
fi

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
BASELINE_ROLLOUT_DIR=$1
GRAPH_RPO_DELTA_MAX=${GRAPH_RPO_DELTA_MAX:-0.25}
DATA_SEED=${DATA_SEED:-42}
SMOKE_SCRIPT=${SMOKE_SCRIPT:-scripts/smoke_train_bc_ctxgraph_8b_graphrpo_qwen3_8b_4node_judge_audit.sh}

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"
if [ ! -d "$BASELINE_ROLLOUT_DIR" ]; then
  echo "ERROR: baseline rollout directory does not exist: $BASELINE_ROLLOUT_DIR"
  exit 1
fi

AUDIT_OUTPUT=$(mktemp)
trap 'rm -f "$AUDIT_OUTPUT"' EXIT
python scripts/audit_counterfactual_graph_credit.py "$BASELINE_ROLLOUT_DIR" --backend reference_answer_likelihood --output "$AUDIT_OUTPUT" --fail-on-integrity-error --fail-on-semantic-noop --require-nonzero-delta
GRAPH_RPO_DELTA_SCALE=$(python -c "import json,sys; summary=json.load(open(sys.argv[1], encoding='utf-8'))['summary']; stats=summary['raw_abs_delta_distribution']; assert stats.get('count', 0) > 0, 'no scored raw deltas available for calibration'; value=max(1.0, float(stats['p80']) / float(sys.argv[2])); print(f'{value:.10g}')" "$AUDIT_OUTPUT" "$GRAPH_RPO_DELTA_MAX")
SCALE_TAG=${GRAPH_RPO_DELTA_SCALE//./p}
RUN_TAG="graphrpo_ref_scaled_smoke_scale${SCALE_TAG}_seed${DATA_SEED}"

echo "Calibrated GraphRPO delta scale: $GRAPH_RPO_DELTA_SCALE"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ALLOCATED_NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | wc -l)
  if [ "$ALLOCATED_NODES" -ne 4 ]; then
    echo "ERROR: direct GraphRPO smoke requires a four-node allocation; got $ALLOCATED_NODES"
    exit 1
  fi
  echo "Running one-step smoke directly inside Slurm allocation $SLURM_JOB_ID"
  export RUN_TAG GRAPH_RPO_DELTA_SCALE GRAPH_RPO_DELTA_MAX DATA_SEED
  export GRAPH_RPO_ALPHA=0.1
  export GRAPH_RPO_AUDIT_MAX_CLIP_RATE=1.0
  bash "$SMOKE_SCRIPT"
  exit 0
fi

echo "No active Slurm allocation detected; submitting one-step smoke"
SUBMISSION=$(sbatch --parsable --export="ALL,RUN_TAG=$RUN_TAG,GRAPH_RPO_ALPHA=0.1,GRAPH_RPO_DELTA_SCALE=$GRAPH_RPO_DELTA_SCALE,GRAPH_RPO_DELTA_MAX=$GRAPH_RPO_DELTA_MAX,GRAPH_RPO_AUDIT_MAX_CLIP_RATE=1.0,DATA_SEED=$DATA_SEED" "$SMOKE_SCRIPT")
JOB_ID=${SUBMISSION%%;*}
if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: sbatch did not return a valid job ID: $SUBMISSION"
  exit 1
fi

echo "Submitted GraphRPO delta-scale smoke: $JOB_ID"
echo "  scale: $GRAPH_RPO_DELTA_SCALE"
echo "  seed:  $DATA_SEED"
