#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
CONDA_ROOT=${CONDA_ROOT:-/work/09281/chc_1996/vista/miniconda3}
INPUT=${INPUT:-$SCRATCH/datasets/MiroVerse-v0.1/jsonl_sft/MiroVerse-MuSiQue.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/contextgraph_sft/miroverse_policy_qwen3_8b}
TRAIN_FILE=${TRAIN_FILE:-$OUTPUT_DIR/contextgraph_sft_train.parquet}
VAL_FILE=${VAL_FILE:-$OUTPUT_DIR/contextgraph_sft_validation.parquet}
MANIFEST=${MANIFEST:-$OUTPUT_DIR/manifest.json}
AUDIT=${AUDIT:-$OUTPUT_DIR/source_audit.json}
MAX_SAMPLES=${MAX_SAMPLES:-0}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.05}
RUN_PREFLIGHT=${RUN_PREFLIGHT:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
TOKENIZER_PATH=${TOKENIZER_PATH:-$SCRATCH/contextgraph_sft_models/957946_miroverse_qwen3_8b_lora32_4k_merged}
MAX_LENGTH=${MAX_LENGTH:-32768}
LOSS_MASK_MODE=${LOSS_MASK_MODE:-chatml}

cd "$PROJECT_ROOT"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate cxtgraph
mkdir -p "$OUTPUT_DIR"

if [[ "$PREFLIGHT_ONLY" != "1" ]]; then
  python -u scripts/inspect_miroverse_policy_source.py --input "$INPUT" --output "$AUDIT" --max-samples "$MAX_SAMPLES"
  python -u scripts/prepare_miroverse_contextgraph_policy_sft.py --input "$INPUT" --output "$TRAIN_FILE" --validation-output "$VAL_FILE" --manifest "$MANIFEST" --validation-fraction "$VALIDATION_FRACTION" --max-samples "$MAX_SAMPLES" --source-subset MiroVerse-MuSiQue
fi

if [[ "$RUN_PREFLIGHT" == "1" || "$PREFLIGHT_ONLY" == "1" ]]; then
  test -s "$TRAIN_FILE" || { echo "ERROR: missing policy train parquet: $TRAIN_FILE"; exit 2; }
  test -s "$VAL_FILE" || { echo "ERROR: missing policy validation parquet: $VAL_FILE"; exit 2; }
  python -u scripts/check_contextgraph_sft_data.py --data "$TRAIN_FILE" --tokenizer "$TOKENIZER_PATH" --max-length "$MAX_LENGTH" --loss-mask-mode "$LOSS_MASK_MODE" --all-samples
  python -u scripts/check_contextgraph_sft_data.py --data "$VAL_FILE" --tokenizer "$TOKENIZER_PATH" --max-length "$MAX_LENGTH" --loss-mask-mode "$LOSS_MASK_MODE" --all-samples
fi

echo "MiroVerse ContextGraph policy SFT complete: $OUTPUT_DIR"
