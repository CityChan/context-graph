#!/bin/bash
#SBATCH -J cg-alf-zs
#SBATCH -o logs/cg-alf-zs.%j.out
#SBATCH -e logs/cg-alf-zs.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# Submit from a Vista login node. The delegated evaluator also remains usable
# interactively from an existing four-node idev allocation.
set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"

exec bash scripts/eval_alfworld_ctxgraph_8b_4node_zeroshot_idev.sh
