#!/bin/bash
#SBATCH -J gen-miroverse-cg-controller
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/gen-miroverse-cg-controller.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/gen-miroverse-cg-controller.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A AST24021

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID}_miroverse_controller_full}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/miroverse_controller_full/$RUN_TAG}

export PROJECT_ROOT RUN_TAG ARTIFACT_ROOT
export MAX_SAMPLES=0

cd "$PROJECT_ROOT"
exec bash scripts/smoke_miroverse_controller_sft_deepseek_v4_4node_idev.sh
