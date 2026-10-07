#!/bin/bash
#SBATCH -J graphrpo-em
#SBATCH -o logs/graphrpo-em.%j.out
#SBATCH -e logs/graphrpo-em.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -A AST24021

# One E attempt then one M attempt, sharing the same actor and optimizer.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export GRAPH_RPO_ALTERNATING_ROLES=True
export GRAPH_RPO_EXECUTOR_SAMPLES=${GRAPH_RPO_EXECUTOR_SAMPLES:-2}
export GRAPH_RPO_CONTINUATION_SAMPLES=${GRAPH_RPO_CONTINUATION_SAMPLES:-2}
export GRAPH_RPO_CONTINUATION_CONCURRENCY=${GRAPH_RPO_CONTINUATION_CONCURRENCY:-2}
export RUN_TAG=${RUN_TAG:-graphrpo_alternating_em}
exec bash "$PROJECT_ROOT/scripts/train_bcp_graphrpo_continuation.sh"
