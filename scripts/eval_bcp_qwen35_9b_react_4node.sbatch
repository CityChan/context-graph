#!/bin/bash
#SBATCH --job-name=bcp-react-9b
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/bcp-qwen35-9b-react.%j.out
#SBATCH --error=logs/bcp-qwen35-9b-react.%j.err

# Submit from the Vista login node after creating logs/.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"

# Pin the requested benchmark so a previously exported GAIA setting cannot leak in.
export BENCHMARK=bcp
export DATA_PATH="$PROJECT_ROOT/data/bc_test.parquet"
export SAMPLES=-1
unset RUN_ROOT EXPECTED_JOB_ID

# The shared launcher resolves the existing 9B checkpoint from scratch offline,
# starts one retriever and three model replicas, and writes a fresh suite directory.
exec bash scripts/eval_bcp_qwen35_9b_4node_idev.sh react
