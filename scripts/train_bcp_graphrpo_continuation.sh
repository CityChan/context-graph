#!/bin/bash
#SBATCH -J graphrpo-continuation
#SBATCH -o logs/graphrpo-continuation.%j.out
#SBATCH -e logs/graphrpo-continuation.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -A AST24021

# Opt-in M-only pilot. Reuse the existing deployment and checkpoint setup.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export GRAPH_RPO_CREDIT_BACKEND=old_policy_continuation
export GRAPH_RPO_CONTINUATION_SAMPLES=${GRAPH_RPO_CONTINUATION_SAMPLES:-4}
export GRAPH_RPO_CONTINUATION_CHECKPOINT=${GRAPH_RPO_CONTINUATION_CHECKPOINT:-1}
export ROLLOUT_N=${ROLLOUT_N:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-2}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-4}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TEST_FREQ=${TEST_FREQ:-1}
export SAVE_FREQ=${SAVE_FREQ:-1}
export RUN_TAG=${RUN_TAG:-graphrpo_continuation_m_only}
exec bash "$PROJECT_ROOT/scripts/train_bc_ctxgraph_8b_graphrpo_5node_48h.sh"
