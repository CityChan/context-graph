#!/bin/bash
#SBATCH --job-name=bcp-9b-cg-64k
#SBATCH --partition=gh
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/bcp-9b-cg-64k.%j.out
#SBATCH --error=logs/bcp-9b-cg-64k.%j.err
set -euo pipefail
export BCP_TRAIN_PROFILE=contextgraph_64k_paper_batch
export SMOKE_TEST=0 PREFLIGHT_ONLY=0
exec bash scripts/train_bcp_qwen35_9b_50step.sh contextgraph
