#!/bin/bash
#SBATCH --job-name=bcp-9b-cg-4x4
#SBATCH --partition=gh
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/bcp-9b-cg-4x4.%j.out
#SBATCH --error=logs/bcp-9b-cg-4x4.%j.err
set -euo pipefail
export BCP_TRAIN_PROFILE=contextgraph_32k_4x4_batch BCP_TRAIN_TOPOLOGY=full
export SMOKE_TEST=0 PREFLIGHT_ONLY=0
case "${1:-train}" in
  train) export BCP_GRAPH_RPO_SMOKE=0 ;;
  smoke) export BCP_GRAPH_RPO_SMOKE=1 ;;
  *) echo "Usage: sbatch $0 [train|smoke]" >&2; exit 2 ;;
esac
exec bash scripts/train_bcp_qwen35_9b_50step.sh contextgraph
