#!/bin/bash
#SBATCH -J sft-miroverse-qwen3-8b
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/sft-miroverse-qwen3-8b.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/sft-miroverse-qwen3-8b.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A AST24021

set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export FORMAL_DATA_DIR=${FORMAL_DATA_DIR:-$SCRATCH/contextgraph_sft/miroverse_controller_qwen3_8b}
export MAX_LENGTH=4096
export DATA_PREFLIGHT_TIMEOUT=${DATA_PREFLIGHT_TIMEOUT:-3600}
export FORMAL_RUN_TAG=${FORMAL_RUN_TAG:-${SLURM_JOB_ID}_miroverse_qwen3_8b_4k_fullparam}

cd "$PROJECT_ROOT"
exec bash scripts/train_contextgraph_sft_qwen3_8b_32k_4node_idev.sh
