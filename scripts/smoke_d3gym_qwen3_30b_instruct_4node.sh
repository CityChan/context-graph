#!/bin/bash
# Run one official D3-Gym task with Qwen3-30B-A3B-Instruct-2507 inside an
# existing four-node Vista GH idev allocation. This is a compatibility smoke,
# not a benchmark result.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"

export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
export D3GYM_MODE=smoke
export D3GYM_METHOD=${D3GYM_METHOD:-react}
export D3GYM_RUNTIME=${D3GYM_RUNTIME:-apptainer}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
export CONDA_ENV_NAME=${CONDA_ENV_NAME:-cxtgraph}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export ROLLOUT_N=1
export SAB_VAL_MAX_SAMPLES=1
export SAB_TRAIN_MAX_SAMPLES=4
export SAB_PROMPT_LENGTH=${SAB_PROMPT_LENGTH:-16384}
export SAB_RESPONSE_LENGTH=${SAB_RESPONSE_LENGTH:-24576}
export SAB_MAX_TOKEN_LEN_PER_GPU=${SAB_MAX_TOKEN_LEN_PER_GPU:-40960}
export SAB_VAL_MAX_TURN=${SAB_VAL_MAX_TURN:-32}
export SAB_TURN_MAX_NEW_TOKENS=${SAB_TURN_MAX_NEW_TOKENS:-2048}
export SAB_DISABLE_WANDB=${SAB_DISABLE_WANDB:-1}
export QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-False}

echo "D3-Gym Qwen3-30B smoke: nodes=$EXPECTED_NUM_NODES method=$D3GYM_METHOD runtime=$D3GYM_RUNTIME"
bash scripts/run_d3gym_30b_instruct_8node.sh
