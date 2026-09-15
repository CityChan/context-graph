#!/bin/bash

# Calibrate frozen-reference GraphRPO from a prior rollout and submit a matched
# no-credit control and scaled-credit treatment. Both jobs retain identical
# data, rollout, validation, and optimizer settings.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: bash scripts/submit_matched_graphrpo_delta_scale_20step.sh BASELINE_ROLLOUT_DIR"
  exit 2
fi

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
BASELINE_ROLLOUT_DIR=$1
GRAPH_RPO_DELTA_MAX=${GRAPH_RPO_DELTA_MAX:-0.25}
GRAPH_RPO_TARGET_CLIP_RATE=${GRAPH_RPO_TARGET_CLIP_RATE:-0.2}
DATA_SEED=${DATA_SEED:-42}
PILOT_SCRIPT=${PILOT_SCRIPT:-scripts/pilot_train_bc_ctxgraph_8b_graphrpo_ref_zeroshot_4node_20step_val.sh}

cd "$PROJECT_ROOT"
if [ ! -d "$BASELINE_ROLLOUT_DIR" ]; then
  echo "ERROR: baseline rollout directory does not exist: $BASELINE_ROLLOUT_DIR"
  exit 1
fi
if [ "$GRAPH_RPO_TARGET_CLIP_RATE" != "0.2" ]; then
  echo "ERROR: this p80 calibrator currently requires GRAPH_RPO_TARGET_CLIP_RATE=0.2"
  exit 1
fi

AUDIT_OUTPUT=$(mktemp)
trap 'rm -f "$AUDIT_OUTPUT"' EXIT
python scripts/audit_counterfactual_graph_credit.py "$BASELINE_ROLLOUT_DIR" --backend reference_answer_likelihood --output "$AUDIT_OUTPUT" --fail-on-integrity-error --fail-on-semantic-noop --require-nonzero-delta
GRAPH_RPO_DELTA_SCALE=$(python -c "import json,sys; summary=json.load(open(sys.argv[1], encoding='utf-8'))['summary']; stats=summary['raw_abs_delta_distribution']; assert stats.get('count', 0) > 0, 'no scored raw deltas available for calibration'; value=max(1.0, float(stats['p80']) / float(sys.argv[2])); print(f'{value:.10g}')" "$AUDIT_OUTPUT" "$GRAPH_RPO_DELTA_MAX")

echo "Calibrated GraphRPO delta scale: $GRAPH_RPO_DELTA_SCALE"
echo "Target: at most 20% of scored edits clipped at delta_max=$GRAPH_RPO_DELTA_MAX"

CONTROL_TAG="graphrpo_ref_control_a0_scale${GRAPH_RPO_DELTA_SCALE}_seed${DATA_SEED}_20step_val"
TREATMENT_TAG="graphrpo_ref_scaled_a01_scale${GRAPH_RPO_DELTA_SCALE}_seed${DATA_SEED}_20step_val"
CONTROL_SUBMISSION=$(sbatch --parsable --export="ALL,RUN_TAG=$CONTROL_TAG,GRAPH_RPO_ALPHA=0.0,GRAPH_RPO_DELTA_SCALE=$GRAPH_RPO_DELTA_SCALE,GRAPH_RPO_DELTA_MAX=$GRAPH_RPO_DELTA_MAX,GRAPH_RPO_AUDIT_MAX_CLIP_RATE=$GRAPH_RPO_TARGET_CLIP_RATE,DATA_SEED=$DATA_SEED" "$PILOT_SCRIPT")
TREATMENT_SUBMISSION=$(sbatch --parsable --export="ALL,RUN_TAG=$TREATMENT_TAG,GRAPH_RPO_ALPHA=0.1,GRAPH_RPO_DELTA_SCALE=$GRAPH_RPO_DELTA_SCALE,GRAPH_RPO_DELTA_MAX=$GRAPH_RPO_DELTA_MAX,GRAPH_RPO_AUDIT_MAX_CLIP_RATE=$GRAPH_RPO_TARGET_CLIP_RATE,DATA_SEED=$DATA_SEED" "$PILOT_SCRIPT")
CONTROL_JOB_ID=${CONTROL_SUBMISSION%%;*}
TREATMENT_JOB_ID=${TREATMENT_SUBMISSION%%;*}

if ! [[ "$CONTROL_JOB_ID" =~ ^[0-9]+$ && "$TREATMENT_JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: sbatch did not return valid job IDs: control=$CONTROL_SUBMISSION treatment=$TREATMENT_SUBMISSION"
  exit 1
fi

echo "Submitted matched GraphRPO jobs"
echo "  control:   $CONTROL_JOB_ID (alpha=0.0)"
echo "  treatment: $TREATMENT_JOB_ID (alpha=0.1)"
echo "  scale:     $GRAPH_RPO_DELTA_SCALE"
echo "  seed:      $DATA_SEED"
