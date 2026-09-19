#!/bin/bash
#SBATCH -J cg-alf-sft-full
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/cg-alf-sft-full.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/cg-alf-sft-full.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A AST24021

# Submit from a Vista login node. MAX_SAMPLES=0 evaluates every supported task
# in ALFWorld valid_seen + valid_unseen. Override MAX_SAMPLES=32 for the
# protocol-matched FoldAgent comparison.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export PROJECT_ROOT
export MODEL_PATH=${MODEL_PATH:-/scratch/09281/chc_1996/contextgraph_sft_models/998826_miroverse_qwen3_8b_lora32_4k_merged}
export MAX_SAMPLES=${MAX_SAMPLES:-0}
export START_INDEX=${START_INDEX:-0}
export DATA_SEED=${DATA_SEED:-42}
export NUM_WORKERS=${NUM_WORKERS:-4}
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID}_alfworld_controller_sft_full_seed${DATA_SEED}}

cd "$PROJECT_ROOT"
exec bash scripts/eval_alfworld_ctxgraph_qwen3_8b_controller_sft_1node_idev.sh
