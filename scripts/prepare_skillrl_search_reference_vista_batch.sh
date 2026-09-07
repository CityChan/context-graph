#!/bin/bash
#SBATCH -J skillrl-prep
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/skillrl-prep.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/skillrl-prep.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A AST24021

# Batch wrapper for all SkillRL/Search-R1 environment setup, downloads, index
# assembly, corpus decompression, and deterministic diagnostic data sampling.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"

cd "$PROJECT_ROOT"
exec bash scripts/prepare_skillrl_search_reference_vista.sh
