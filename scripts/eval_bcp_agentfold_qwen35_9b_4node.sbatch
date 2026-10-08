#!/bin/bash
#SBATCH --job-name=bcp-agentfold-9b
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=agentfold-9b.%j.out
#SBATCH --error=agentfold-9b.%j.err

# AgentFold-style zero-shot adaptation, not the authors' trained checkpoint.
# Default: 8-task smoke, same 32K context / 100 turns / tools / judge as BC-P.
# SAMPLES=-1 selects all tasks; use the same SAMPLES and SEED for paired methods.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export SAMPLES=${SAMPLES:-8}
exec bash "$PROJECT_ROOT/scripts/eval_bcp_qwen35_9b_4node_idev.sh" agentfold
