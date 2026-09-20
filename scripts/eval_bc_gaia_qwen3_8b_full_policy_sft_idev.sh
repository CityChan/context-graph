#!/bin/bash

# Evaluate the completed MiroVerse full-policy Qwen3-8B SFT model on BC-P and
# GAIA from an existing four- or five-node Vista idev allocation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/09281/chc_1996}

export EVAL_VARIANT=full_policy_sft
export EVAL_MODEL_TAG=miroverse_full_policy_step174
export EVAL_MODEL_PATH=${FULL_POLICY_SFT_MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/miroverse_full_policy_qwen3_8b_bs16_1ep_v1_step174_hf}
export EVAL_CTXGRAPH_PROTOCOL=full_policy

test -s "$EVAL_MODEL_PATH/config.json" || { echo "ERROR: invalid full-policy SFT model: $EVAL_MODEL_PATH" >&2; exit 2; }

exec bash "$PROJECT_ROOT/scripts/eval_bc_gaia_qwen3_8b_base_idev.sh"
