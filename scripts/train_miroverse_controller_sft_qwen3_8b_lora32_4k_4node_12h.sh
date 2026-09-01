#!/bin/bash
#SBATCH -J sft-miroverse-qwen3-8b-lora
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/sft-miroverse-qwen3-8b-lora.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/sft-miroverse-qwen3-8b-lora.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A AST24021

set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export FORMAL_DATA_DIR=${FORMAL_DATA_DIR:-$SCRATCH/contextgraph_sft/miroverse_controller_qwen3_8b}
export MAX_LENGTH=4096
export LORA_RANK=32
export LORA_ALPHA=64
export TRAIN_LR=${TRAIN_LR:-1e-4}
export DATA_PREFLIGHT_TIMEOUT=${DATA_PREFLIGHT_TIMEOUT:-3600}
export FORMAL_RUN_TAG=${FORMAL_RUN_TAG:-${SLURM_JOB_ID}_miroverse_qwen3_8b_lora32_4k}
export FORMAL_CHECKPOINT_ROOT=${FORMAL_CHECKPOINT_ROOT:-$SCRATCH/contextgraph_sft_checkpoints/$FORMAL_RUN_TAG}
export FORMAL_MERGED_MODEL_DIR=${FORMAL_MERGED_MODEL_DIR:-$SCRATCH/contextgraph_sft_models/${FORMAL_RUN_TAG}_base_with_adapter}
FINAL_MODEL_DIR=${FINAL_MODEL_DIR:-$SCRATCH/contextgraph_sft_models/${FORMAL_RUN_TAG}_merged}

cd "$PROJECT_ROOT"
bash scripts/train_contextgraph_sft_qwen3_8b_32k_4node_idev.sh

python -u scripts/merge_contextgraph_lora_hf.py --base-model "$FORMAL_MERGED_MODEL_DIR" --adapter "$FORMAL_MERGED_MODEL_DIR/lora_adapter" --output "$FINAL_MODEL_DIR" --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
printf '%s\n' "$FINAL_MODEL_DIR" > "$FORMAL_CHECKPOINT_ROOT/merged_hf_model_path.txt"
echo "Formal Qwen3-8B MiroVerse LoRA SFT completed: $FINAL_MODEL_DIR"
