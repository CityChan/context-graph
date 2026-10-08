#!/bin/bash
#SBATCH --job-name=stateful-memory-9b
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=stateful-memory-%j.out
#SBATCH --error=stateful-memory-%j.err
set -euo pipefail
# Two-task smoke by default. SAMPLES=-1 selects all tasks after smoke validation.
# Slurm copies this file to its spool, so never resolve siblings from $0 here.
exec bash "${STATEFUL_PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}/scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh" "${1:?benchmark}" "${2:?method}"
