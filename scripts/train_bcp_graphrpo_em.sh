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

# Compatibility entry point. New submissions use train_bcp_em_rpo.sh.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export RUN_TAG=${RUN_TAG:-graphrpo_alternating_em}
exec bash "$PROJECT_ROOT/scripts/train_bcp_em_rpo.sh"
