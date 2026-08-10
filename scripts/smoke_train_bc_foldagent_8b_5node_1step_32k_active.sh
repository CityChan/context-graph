#!/bin/bash
#SBATCH -J smoke-bc-8b-fa-train
#SBATCH -o logs/smoke-bc-8b-fa-train.%j.out
#SBATCH -e logs/smoke-bc-8b-fa-train.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

set -euo pipefail

# One real FoldGRPO optimizer step through the production FoldAgent entrypoint.
# Four training samples x two rollouts = eight trajectories; ppo_mini=2 per
# trainer rank x four ranks = the same global batch of eight.
export RUN_TAG=smoke_1step_32k_active
export PROMPT_LENGTH=8192
export RESPONSE_LENGTH=24576
export CONTEXT_LENGTH=32768
export TRAIN_BATCH_SIZE=4
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
