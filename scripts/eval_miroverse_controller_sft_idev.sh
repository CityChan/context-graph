#!/bin/bash
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_PATH=${MODEL_PATH:-$SCRATCH/contextgraph_sft_models/957946_miroverse_qwen3_8b_lora32_4k_merged}
VALIDATION_FILE=${VALIDATION_FILE:-$SCRATCH/contextgraph_sft/miroverse_controller_qwen3_8b/contextgraph_sft_validation.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SCRATCH/cgeval}
GUIDED_DECODING=${GUIDED_DECODING:-0}
if [ "$GUIDED_DECODING" = "1" ]; then
  DECODING_TAG=guided
  DECODING_FLAG=--guided-decoding
else
  DECODING_TAG=unguided
  DECODING_FLAG=--no-guided-decoding
fi
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-local}_miroverse_controller_$DECODING_TAG}
OUTPUT_JSON=$OUTPUT_ROOT/$RUN_TAG.json
LOG_PATH=$PROJECT_ROOT/logs/controller-eval-${SLURM_JOB_ID:-local}-$DECODING_TAG.log

test -s "$MODEL_PATH/config.json" || { echo "ERROR: missing merged model: $MODEL_PATH"; exit 2; }
test -s "$VALIDATION_FILE" || { echo "ERROR: missing validation parquet: $VALIDATION_FILE"; exit 2; }

mkdir -p "$OUTPUT_ROOT" "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/contextgraph-vllm-${SLURM_JOB_ID:-$$}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/contextgraph-inductor-${SLURM_JOB_ID:-$$}}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/contextgraph-triton-${SLURM_JOB_ID:-$$}}

echo "Controller-only held-out validation"
echo "Model: $MODEL_PATH"
echo "Data: $VALIDATION_FILE"
echo "Output: $OUTPUT_JSON"

set -o pipefail
python -u scripts/eval_miroverse_controller_sft.py --model "$MODEL_PATH" --data "$VALIDATION_FILE" --output "$OUTPUT_JSON" --max-samples 0 --max-model-len 4096 --max-num-seqs 64 --gpu-memory-utilization 0.85 --enforce-eager "$DECODING_FLAG" 2>&1 | tee "$LOG_PATH"
