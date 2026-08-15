#!/bin/bash
#SBATCH -J train-bc-8b-react-grpo-5n
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/train-bc-8b-react-grpo-5n.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/train-bc-8b-react-grpo-5n.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH -A AST24021

# Paper-aligned vanilla ReAct control:
#   standard GRPO, no fold, no branch, no graph, no consolidation.
#   8K prompt + 32K response, 5 nodes, 50 optimization steps.
set -euo pipefail

export EXPECTED_NUM_NODES=5
export RUN_TAG=react_grpo_5n_50step_32k_active
export ADV_ESTIMATOR=grpo
export PROMPT_LENGTH=8192
export RESPONSE_LENGTH=32768
export CONTEXT_LENGTH=40960
export TRAIN_BATCH_SIZE=32
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_N=8
export TRAIN_LR=1e-6
export SESSION_TIMEOUT=3600
export TOTAL_TRAINING_STEPS=50
export TEST_FREQ=10
export SAVE_FREQ=10

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/train_bc_baseline_8b_4node_24h_v3_32k.sh"
