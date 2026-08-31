#!/bin/bash

# Matched Aug-27 BrowseComp-Plus controller evaluation for the locally merged
# Qwen3-8B ContextGraph SFT model. Run inside the same four-node Vista idev
# allocation used for formal SFT.
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}

# Ignore model/protocol/cache values left over from earlier smoke or training
# runs. The Aug-27 eval artefacts live in the established read-only work cache.
unset MODEL_PATH EXPERIMENT_NAME HF_HOME HF_HUB_CACHE
export HF_HOME=/work/09281/chc_1996/vista/cache
export HF_HUB_CACHE=$HF_HOME/hub
export MODEL_PATH=${SFT_EVAL_MODEL_PATH:-$SCRATCH/contextgraph_sft_models/954050_qwen3_8b_contextgraph_sft_32k_fullparam}
export BC_METHOD=contextgraph
export BC_CTXGRAPH_PROTOCOL=controller
export BC_CONTROLLER_ACTION_POLICY=structural
export BC_EXPERIMENT_MODEL_TAG=qwen3_8b_sft_32k
export BC_CONTEXT_LENGTH=65536
export BC_PROMPT_LENGTH=8192
export BC_RESPONSE_LENGTH=57344
export BC_YARN_FACTOR=2.0
export BC_YARN_ORIGINAL_LENGTH=32768
export BC_FINAL_ANSWER_RESERVE=1024
export BC_VAL_MAX_SAMPLES=-1
export BC_DISABLE_WANDB=1
export EXPECTED_NUM_NODES=4
export EXPERIMENT_NAME=eval_contextgraph_controller_bc_qwen3_8b_sft_32k_4n_65536ctx_$(date +%Y%m%d_%H%M%S)

if [ ! -s "$MODEL_PATH/config.json" ]; then
  echo "ERROR: merged SFT model is missing config.json: $MODEL_PATH"
  exit 2
fi

echo "Matched Aug-27 BrowseComp-Plus SFT evaluation"
echo "Model: $MODEL_PATH"
echo "Protocol: method=$BC_METHOD graph=$BC_CTXGRAPH_PROTOCOL action_policy=$BC_CONTROLLER_ACTION_POLICY context=$BC_CONTEXT_LENGTH samples=all"

exec bash "$PROJECT_ROOT/scripts/eval_bc_baseline_8b_4node_zeroshot.sh"
