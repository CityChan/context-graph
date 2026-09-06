#!/bin/bash
# Submit the corrected paired-counterfactual smoke and make the formal stage-2
# job depend on all signal-quality gates passing.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
MODEL_PATH=${MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/miroverse_full_policy_qwen3_8b_bs16_1ep_v1_step174_hf}

cd "$PROJECT_ROOT"
mkdir -p logs

SMOKE_JOB_ID=$(sbatch --parsable --export=ALL,MODEL_PATH="$MODEL_PATH" scripts/smoke_train_bc_ctxgraph_8b_graphrpo_counterfactual_4node_2step.sh)
FORMAL_JOB_ID=$(sbatch --parsable --dependency=afterok:"$SMOKE_JOB_ID" --export=ALL,MODEL_PATH="$MODEL_PATH" scripts/train_bc_ctxgraph_8b_graphrpo_5node_48h.sh)

echo "Counterfactual signal-gated stage-2 submission complete"
echo "  smoke job: $SMOKE_JOB_ID"
echo "  formal job: $FORMAL_JOB_ID (starts only after smoke passes)"
echo "  model: $MODEL_PATH"
