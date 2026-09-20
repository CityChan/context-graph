#!/bin/bash

# Evaluate the completed MiroVerse full-policy Qwen3-8B SFT model on the same
# fixed 32-episode ALFWorld split used by the Base 8B comparison.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/09281/chc_1996}

export MODEL_PATH=${FULL_POLICY_SFT_MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/miroverse_full_policy_qwen3_8b_bs16_1ep_v1_step174_hf}
export ALFWORLD_EXPERIMENT_PREFIX=full_policy_sft
export ALFWORLD_METHOD_LABEL="ContextGraph full-policy SFT"
export ALFWORLD_STRUCTURED_GRAPH_CONTROLLER=False
export ALFWORLD_CONTROLLER_OWNED_TOOL_FORMATTING=False
export ALFWORLD_CONTROLLER_ACTION_POLICY=balanced

test -s "$MODEL_PATH/config.json" || { echo "ERROR: invalid full-policy SFT model: $MODEL_PATH" >&2; exit 2; }

exec bash "$PROJECT_ROOT/scripts/eval_alfworld_ctxgraph_8b_4node_zeroshot_idev.sh"
