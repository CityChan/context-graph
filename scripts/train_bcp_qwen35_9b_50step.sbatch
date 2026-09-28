#!/bin/bash
#SBATCH --job-name=bcp-9b-rl50
#SBATCH --partition=gh
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/bcp-9b-rl50.%x.%j.out
#SBATCH --error=logs/bcp-9b-rl50.%x.%j.err
set -euo pipefail
exec bash scripts/train_bcp_qwen35_9b_50step.sh "${1:?Expected contextgraph or foldagent}"
