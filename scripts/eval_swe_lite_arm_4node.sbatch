#!/bin/bash
#SBATCH --job-name=swe-lite-arm
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/swe-lite-arm.%j.out
#SBATCH --error=logs/swe-lite-arm.%j.err

# One allocation owns model servers, Apptainer setup, generation and grading.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
method=${1:-${SWE_METHOD:-contextgraph}}
case "$method" in contextgraph|foldagent|react) ;; *) echo 'Expected contextgraph, foldagent or react' >&2; exit 2;; esac
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export SWE_AGENT_ENV=${SWE_AGENT_ENV:-/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP}
export SWE_CONTEXT_LENGTH=${SWE_CONTEXT_LENGTH:-65536}
export SWE_SAMPLES=${SWE_SAMPLES:--1}
cd "$PROJECT_ROOT"
for python in bin/python grading/bin/python preparation/bin/python; do
    [[ -x "$SWE_AGENT_ENV/$python" ]] || { echo "Missing environment: $SWE_AGENT_ENV/$python" >&2; exit 2; }
done
echo "SWE_ARM_BATCH job=$SLURM_JOB_ID method=$method commit=$(git rev-parse HEAD) context=$SWE_CONTEXT_LENGTH samples=$SWE_SAMPLES"
exec bash scripts/run_swe_lite_arm_4node_idev.sh "$method"
