#!/bin/bash
#SBATCH -J train-bc-8b-cg-64k-5n-50s
#SBATCH -o logs/train-bc-8b-cg-64k-5n-50s.%j.out
#SBATCH -e logs/train-bc-8b-cg-64k-5n-50s.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH -A AST24021

set -euo pipefail

# Match FoldAgent's model, data, optimization, sample budget, and 64K active
# context. Only the agent workflow and method-specific process reward differ.
export RUN_TAG=foldgrpo_5n_50step_64k_active
export PROMPT_LENGTH=8192
export RESPONSE_LENGTH=57344
export CONTEXT_LENGTH=65536
export BC_YARN_FACTOR=2.0
export BC_YARN_ORIGINAL_LENGTH=32768
# This fork interprets ppo_mini_batch_size per trainer rank when padding the
# global batch: 32 per rank x 4 trainer ranks = paper-scale global 128.
export PPO_MINI_BATCH_SIZE=32
export TRAIN_LR=1e-6
export USE_KL_LOSS=False
export ACTOR_KL_LOSS_COEF=0.0
export ALGORITHM_KL_COEF=0.0
export CLIP_RATIO_LOW=0.2
export CLIP_RATIO_HIGH=0.28

exec bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
