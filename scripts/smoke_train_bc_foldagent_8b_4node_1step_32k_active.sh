#!/bin/bash

# One real FoldGRPO optimizer step inside an existing four-node idev:
# one search node plus three trainer ranks.

set -euo pipefail

export EXPECTED_NUM_NODES=4
export RUN_TAG=${RUN_TAG:-idev_4n_smoke_1step_32k_active}
export PROMPT_LENGTH=8192
export RESPONSE_LENGTH=24576
export CONTEXT_LENGTH=32768
export TRAIN_BATCH_SIZE=3
export ROLLOUT_N=2
export PPO_MINI_BATCH_SIZE=2
export TOTAL_TRAINING_STEPS=1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=0
export SAVE_FREQ=0
export TRAIN_LR=1e-6
export USE_KL_LOSS=False
export ACTOR_KL_LOSS_COEF=0.0
export ALGORITHM_KL_COEF=0.0
export CLIP_RATIO_LOW=0.2
export CLIP_RATIO_HIGH=0.28

exec bash scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh
