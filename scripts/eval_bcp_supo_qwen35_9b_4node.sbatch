#!/bin/bash
#SBATCH --job-name=bcp-supo-9b
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=supo-9b.%j.out
#SBATCH --error=supo-9b.%j.err

# Paper-based SUPO rollout, zero-shot; not the jointly RL-trained method.
# Existing allocation: bash scripts/eval_bcp_qwen35_9b_4node_idev.sh supo
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export SAMPLES=${SAMPLES:-8}
exec bash "$PROJECT_ROOT/scripts/eval_bcp_qwen35_9b_4node_idev.sh" supo
