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

# Compatibility entry point; retain its Slurm job/log names.
set -euo pipefail
cd "${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}"
exec bash scripts/eval_swe_lite_arm_4node.sbatch foldagent
