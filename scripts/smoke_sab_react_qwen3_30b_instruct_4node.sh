#!/bin/bash
# Run a one-sample Qwen3-30B-A3B-Instruct SAB compatibility smoke inside an
# existing four-node Vista GH idev allocation. This is not a benchmark result.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"

export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
export CONDA_ENV_NAME=${CONDA_ENV_NAME:-cxtgraph}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:-1}
export SAB_TRAIN_MAX_SAMPLES=${SAB_TRAIN_MAX_SAMPLES:-4}
export SAB_PROMPT_LENGTH=${SAB_PROMPT_LENGTH:-8192}
export SAB_RESPONSE_LENGTH=${SAB_RESPONSE_LENGTH:-2048}
export SAB_MAX_TOKEN_LEN_PER_GPU=${SAB_MAX_TOKEN_LEN_PER_GPU:-10240}
export SAB_VAL_MAX_TURN=${SAB_VAL_MAX_TURN:-4}
export SAB_TURN_MAX_NEW_TOKENS=${SAB_TURN_MAX_NEW_TOKENS:-512}
export SAB_DISABLE_WANDB=${SAB_DISABLE_WANDB:-1}
export QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-False}

echo "Qwen3-30B-A3B-Instruct SAB ReAct smoke: nodes=$EXPECTED_NUM_NODES samples=$SAB_VAL_MAX_SAMPLES prompt=$SAB_PROMPT_LENGTH response=$SAB_RESPONSE_LENGTH"
bash scripts/eval_sab_react_30b_instruct_8node_smoke.sh
