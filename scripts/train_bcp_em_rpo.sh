#!/bin/bash
#SBATCH -J em-rpo
#SBATCH -o logs/em-rpo.%j.out
#SBATCH -e logs/em-rpo.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -A AST24021

# EM-RPO: one E attempt then one M attempt, sharing actor and optimizer.
# Keep GraphRPO config keys for compatibility with saved runs and checkpoints.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export GRAPH_RPO_ALTERNATING_ROLES=True
export GRAPH_RPO_EXECUTOR_SAMPLES=${GRAPH_RPO_EXECUTOR_SAMPLES:-2}
export GRAPH_RPO_CONTINUATION_SAMPLES=${GRAPH_RPO_CONTINUATION_SAMPLES:-2}
export GRAPH_RPO_CONTINUATION_CONCURRENCY=${GRAPH_RPO_CONTINUATION_CONCURRENCY:-2}
export RUN_TAG=${RUN_TAG:-em_rpo}
exec bash "$PROJECT_ROOT/scripts/train_bcp_graphrpo_continuation.sh"
