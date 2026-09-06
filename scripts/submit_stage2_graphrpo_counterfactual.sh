#!/bin/bash
# Submit the corrected paired-counterfactual smoke and make the formal stage-2
# job depend on all signal-quality gates passing.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
MODEL_PATH=${MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/miroverse_full_policy_qwen3_8b_bs16_1ep_v1_step174_hf}

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: sbatch submission must run on a Vista login node, not inside Slurm job $SLURM_JOB_ID"
  echo "       Exit the compute-node shell, then rerun this submitter from the login node."
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p logs

if ! SMOKE_SUBMISSION=$(sbatch --parsable --export=ALL,MODEL_PATH="$MODEL_PATH" scripts/smoke_train_bc_ctxgraph_8b_graphrpo_counterfactual_4node_2step.sh 2>&1); then
  echo "ERROR: failed to submit the counterfactual smoke"
  echo "$SMOKE_SUBMISSION"
  exit 1
fi
if [[ ! "$SMOKE_SUBMISSION" =~ ^[0-9]+(;[A-Za-z0-9_.-]+)?$ ]]; then
  echo "ERROR: sbatch did not return a valid smoke job ID"
  echo "$SMOKE_SUBMISSION"
  exit 1
fi
SMOKE_JOB_ID=${SMOKE_SUBMISSION%%;*}

if ! FORMAL_SUBMISSION=$(sbatch --parsable --dependency=afterok:"$SMOKE_JOB_ID" --export=ALL,MODEL_PATH="$MODEL_PATH" scripts/train_bc_ctxgraph_8b_graphrpo_5node_48h.sh 2>&1); then
  echo "ERROR: smoke job $SMOKE_JOB_ID was submitted, but formal-job submission failed"
  echo "$FORMAL_SUBMISSION"
  exit 1
fi
if [[ ! "$FORMAL_SUBMISSION" =~ ^[0-9]+(;[A-Za-z0-9_.-]+)?$ ]]; then
  echo "ERROR: smoke job $SMOKE_JOB_ID was submitted, but sbatch returned no valid formal job ID"
  echo "$FORMAL_SUBMISSION"
  exit 1
fi
FORMAL_JOB_ID=${FORMAL_SUBMISSION%%;*}

echo "Counterfactual signal-gated stage-2 submission complete"
echo "  smoke job: $SMOKE_JOB_ID"
echo "  formal job: $FORMAL_JOB_ID (starts only after smoke passes)"
echo "  model: $MODEL_PATH"
