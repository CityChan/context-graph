#!/bin/bash
#SBATCH --job-name=swe-lite-foldagent
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/swe-lite-foldagent.%j.out
#SBATCH --error=logs/swe-lite-foldagent.%j.err

# Same two-pair workflow as idev; Slurm owns the allocation and captures startup.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export SWE_AGENT_ENV=${SWE_AGENT_ENV:-/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP}
export SWE_CONTEXT_LENGTH=${SWE_CONTEXT_LENGTH:-65536}
export SWE_SAMPLES=${SWE_SAMPLES:--1}
cd "$PROJECT_ROOT"
echo "SWE_FOLDAGENT_BATCH job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) context=$SWE_CONTEXT_LENGTH samples=$SWE_SAMPLES"
exec bash scripts/run_swe_lite_arm_4node_idev.sh foldagent
