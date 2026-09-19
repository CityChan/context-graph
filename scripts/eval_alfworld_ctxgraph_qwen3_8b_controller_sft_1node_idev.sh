#!/bin/bash
# Single-GH200 end-to-end ALFWorld @real diagnostic for the merged MiroVerse
# controller-SFT checkpoint. The checkpoint drives both environment actions
# and graph-controller checkpoints; this is not an isolated controller swap.
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this inside a Vista idev allocation"
  exit 1
fi

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_PATH=${MODEL_PATH:-$SCRATCH/contextgraph_sft_models/998826_miroverse_qwen3_8b_lora32_4k_merged}
MODEL_ID=${MODEL_ID:-contextgraph-qwen3-8b-controller-sft}
SERVER_PORT=${SERVER_PORT:-18000}
MAX_SAMPLES=${MAX_SAMPLES:-8}
START_INDEX=${START_INDEX:-0}
DATA_SEED=${DATA_SEED:-42}
MAX_TURN=${MAX_TURN:-60}
CONSOLIDATION_INTERVAL=${CONSOLIDATION_INTERVAL:-5}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-128}
PROMPT_LENGTH=${PROMPT_LENGTH:-4096}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-12288}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-local}_alfworld_controller_sft}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/cgeval/$RUN_TAG}
RAW_OUTPUT_DIR=$ARTIFACT_ROOT/raw
VLLM_LOG=$ARTIFACT_ROOT/vllm.log

if ! [[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_SAMPLES must be a positive integer"
  exit 2
fi
if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]]; then
  echo "ERROR: START_INDEX must be a non-negative integer"
  exit 2
fi
test -s "$MODEL_PATH/config.json" || { echo "ERROR: invalid MODEL_PATH=$MODEL_PATH"; exit 2; }

mkdir -p "$ARTIFACT_ROOT" "$RAW_OUTPUT_DIR" "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph

export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/contextgraph-vllm-${SLURM_JOB_ID:-$$}}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/contextgraph-inductor-${SLURM_JOB_ID:-$$}}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/contextgraph-triton-${SLURM_JOB_ID:-$$}}
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export QWEN_ENABLE_THINKING=False

test -d "$ALFWORLD_DATA/json_2.1.1" || { echo "ERROR: ALFWorld data missing under $ALFWORLD_DATA"; exit 2; }

python scripts/make_alfworld_data.py \
  --mode real \
  --n_train 16 \
  --n_val "$((START_INDEX + MAX_SAMPLES))" \
  --seed "$DATA_SEED" \
  --alfworld_data "$ALFWORLD_DATA"

DATA_PATH=data/alfworld_graph_real_test.parquet
test -s "$DATA_PATH" || { echo "ERROR: missing $DATA_PATH"; exit 2; }

echo "=============================================================="
echo "  ALFWorld controller-SFT end-to-end diagnostic"
echo "  Model: $MODEL_PATH"
echo "  Episodes: $MAX_SAMPLES from index $START_INDEX; seed: $DATA_SEED"
echo "  Controller: structural, interval=$CONSOLIDATION_INTERVAL, pass disabled"
echo "  Output: $RAW_OUTPUT_DIR"
echo "=============================================================="

vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$SERVER_PORT" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 8 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-chunked-prefill \
  --enforce-eager \
  --guided-decoding-backend guidance >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 600); do
  curl --noproxy '*' -fsS "http://127.0.0.1:$SERVER_PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM exited; inspect $VLLM_LOG"; exit 3; }
  sleep 1
done
curl --noproxy '*' -fsS "http://127.0.0.1:$SERVER_PORT/v1/models" >/dev/null || { echo "ERROR: vLLM health timeout"; exit 3; }

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL="http://127.0.0.1:$SERVER_PORT/v1"

python scripts/eval_interactive.py \
  --data-path "$DATA_PATH" \
  --output-dir "$RAW_OUTPUT_DIR" \
  --workflow alfworld_graph \
  --model-name "$MODEL_ID" \
  --tokenizer-name "$MODEL_PATH" \
  --max-samples "$MAX_SAMPLES" \
  --start-index "$START_INDEX" \
  --num-workers 1 \
  --prompt-length "$PROMPT_LENGTH" \
  --response-length "$RESPONSE_LENGTH" \
  --max-turn "$MAX_TURN" \
  --consolidation-interval "$CONSOLIDATION_INTERVAL" \
  --structured-graph-controller \
  --controller-action-policy structural \
  --no-controller-allow-pass \
  --inject-graph-state-after-action \
  --max-session 3 \
  --branch-len 2048 \
  --turn-max-new-tokens "$TURN_MAX_NEW_TOKENS" \
  --temperature 0.0 \
  --top-p 1.0 \
  --reasoning-effort non-thinking \
  --save-messages

RESULT_FILE=$(find "$RAW_OUTPUT_DIR" -maxdepth 1 -name 'interactive_results_*.json' -type f | sort | tail -n 1)
test -n "$RESULT_FILE" || { echo "ERROR: evaluator produced no result JSON"; exit 3; }
python scripts/validate_contextgraph_traces.py "$RESULT_FILE"
echo "ALFWorld controller-SFT diagnostic complete: $RESULT_FILE"
