#!/bin/bash
# Four-node, validation-only ReAct baseline smoke on official ALFWorld.
set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"

export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}

python scripts/make_alfworld_data.py \
  --mode real \
  --n_train 16 \
  --n_val 4 \
  --alfworld_data "$ALFWORLD_DATA"

export WANDB_MODE=disabled
export ALFWORLD_DISABLE_WANDB=1
export ALFWORLD_TRAIN_MODULE=scripts.train_graph
export ALFWORLD_AGENT_LOOP=react_agent
export ALFWORLD_WORKFLOW=alfworld
export ALFWORLD_PROCESS_REWARD='[flat]'
export ALFWORLD_DATA_VARIANT=alfworld
export ALFWORLD_VAL_ONLY=True
export ALFWORLD_VAL_BEFORE_TRAIN=True
export ALFWORLD_VAL_MAX_SAMPLES=${ALFWORLD_VAL_MAX_SAMPLES:-4}
export ALFWORLD_TRAIN_MAX_SAMPLES=${ALFWORLD_TRAIN_MAX_SAMPLES:-16}
export ALFWORLD_TOTAL_STEPS=1
export ALFWORLD_TRAIN_BATCH_SIZE=4
export ALFWORLD_PPO_MINI_BATCH_SIZE=4
export ALFWORLD_ROLLOUT_N=1
export ALFWORLD_MAX_TURN=${ALFWORLD_MAX_TURN:-30}
export ALFWORLD_VAL_MAX_TURN=${ALFWORLD_VAL_MAX_TURN:-30}
export ALFWORLD_TURN_MAX_NEW_TOKENS=${ALFWORLD_TURN_MAX_NEW_TOKENS:-128}
export ALFWORLD_PROMPT_LENGTH=${ALFWORLD_PROMPT_LENGTH:-4096}
export ALFWORLD_RESPONSE_LENGTH=${ALFWORLD_RESPONSE_LENGTH:-8192}
export ALFWORLD_MAX_TOKEN_LEN_PER_GPU=${ALFWORLD_MAX_TOKEN_LEN_PER_GPU:-12288}

exec bash scripts/train_alfworld_ctxgraph_8b_4node_30step.sh
