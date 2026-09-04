#!/bin/bash
# Train and calibrate a pilot GraphRPO evaluator from raw pre-SFT trajectories.
# This reuses both task-success and task-failure graph traces; curated SFT
# Parquet files are intentionally not used because they are success-filtered.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
RAW_SFT_ROOT=${RAW_SFT_ROOT:-$SCRATCH_ROOT/contextgraph_sft}
GRAPH_EVALUATOR_BASE_MODEL=${GRAPH_EVALUATOR_BASE_MODEL:-Qwen/Qwen3-0.6B}
GRAPH_EVALUATOR_MAX_RAW_FILES=${GRAPH_EVALUATOR_MAX_RAW_FILES:-32}
GRAPH_EVALUATOR_MAX_QUESTIONS=${GRAPH_EVALUATOR_MAX_QUESTIONS:-256}
GRAPH_EVALUATOR_MAX_LENGTH=${GRAPH_EVALUATOR_MAX_LENGTH:-2048}
GRAPH_EVALUATOR_EPOCHS=${GRAPH_EVALUATOR_EPOCHS:-1}
GRAPH_EVALUATOR_BATCH_SIZE=${GRAPH_EVALUATOR_BATCH_SIZE:-1}
GRAPH_EVALUATOR_GRAD_ACCUM=${GRAPH_EVALUATOR_GRAD_ACCUM:-8}
GRAPH_EVALUATOR_VALIDATION_FRACTION=${GRAPH_EVALUATOR_VALIDATION_FRACTION:-0.2}
GRAPH_EVALUATOR_SEED=${GRAPH_EVALUATOR_SEED:-42}
GRAPH_EVALUATOR_PROBE_PORT=${GRAPH_EVALUATOR_PROBE_PORT:-19002}
RUN_TS=$(date +%Y%m%d_%H%M%S)
DATA_DIR=${GRAPH_EVALUATOR_DATA_ROOT:-$SCRATCH_ROOT/context-graph-evaluator-data/sft_raw_pilot_$RUN_TS}
MODEL_DIR=${GRAPH_EVALUATOR_MODEL_ROOT:-$SCRATCH_ROOT/context-graph-evaluators/sft_raw_pilot_qwen3_0p6b_$RUN_TS}
TRAIN_LOG="$MODEL_DIR/train.log"
SERVER_LOG="$MODEL_DIR/server_probe.log"

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}

if [ ! -d "$RAW_SFT_ROOT" ]; then
  echo "ERROR: raw SFT root does not exist: $RAW_SFT_ROOT"
  exit 1
fi

RAW_SFT_REAL=$(realpath -m "$RAW_SFT_ROOT")
DATA_REAL=$(realpath -m "$DATA_DIR")
MODEL_REAL=$(realpath -m "$MODEL_DIR")
for OUTPUT_PATH in "$DATA_REAL" "$MODEL_REAL"; do
  case "$OUTPUT_PATH" in
    "$RAW_SFT_REAL"|"$RAW_SFT_REAL"/*)
      echo "ERROR: evaluator outputs must not be written inside raw SFT data: $OUTPUT_PATH"
      exit 1
      ;;
  esac
done
if [ "$DATA_REAL" = "$MODEL_REAL" ]; then
  echo "ERROR: evaluator data and model roots must be different directories."
  exit 1
fi
mkdir -p "$DATA_DIR" "$MODEL_DIR"

mapfile -t RAW_RESULTS < <(find "$RAW_SFT_ROOT" -type f \( -name 'interactive_results_*.json' -o -name 'gaia_results_*.json' \) -size +0c -printf '%T@ %p\n' | sort -nr | head -n "$GRAPH_EVALUATOR_MAX_RAW_FILES" | cut -d' ' -f2-)
if [ "${#RAW_RESULTS[@]}" -eq 0 ]; then
  echo "ERROR: no raw pre-SFT result JSON found under $RAW_SFT_ROOT"
  echo "Expected .../raw/interactive_results_*.json or .../raw/gaia_results_*.json"
  exit 1
fi
printf '%s\n' "${RAW_RESULTS[@]}" > "$DATA_DIR/source_files.txt"

MODEL_SOURCE="$GRAPH_EVALUATOR_BASE_MODEL"
if [ ! -d "$MODEL_SOURCE" ]; then
  CACHE_NAME="models--${GRAPH_EVALUATOR_BASE_MODEL//\//--}"
  CACHED_SNAPSHOT=$(find "$HF_HUB_CACHE/$CACHE_NAME/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1) || true
  if [ -n "$CACHED_SNAPSHOT" ] && [ -s "$CACHED_SNAPSHOT/config.json" ]; then
    MODEL_SOURCE="$CACHED_SNAPSHOT"
  else
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
  fi
fi

echo "=============================================================="
echo "  GRAPHRPO EVALUATOR PILOT FROM RAW SFT GENERATIONS"
echo "  Raw root:       $RAW_SFT_ROOT"
echo "  Input files:    ${#RAW_RESULTS[@]}"
echo "  Question cap:   $GRAPH_EVALUATOR_MAX_QUESTIONS"
echo "  Base classifier:$MODEL_SOURCE"
echo "  Data output:    $DATA_DIR"
echo "  Model output:   $MODEL_DIR"
echo "=============================================================="

python scripts/prepare_graph_evaluator_data.py "${RAW_RESULTS[@]}" --output-dir "$DATA_DIR" --validation-fraction "$GRAPH_EVALUATOR_VALIDATION_FRACTION" --seed "$GRAPH_EVALUATOR_SEED" --auto-seed-attempts 10000 --require-both-classes --max-questions "$GRAPH_EVALUATOR_MAX_QUESTIONS"

python scripts/train_graph_evaluator.py --train-file "$DATA_DIR/graph_evaluator_train.parquet" --validation-file "$DATA_DIR/graph_evaluator_validation.parquet" --model "$MODEL_SOURCE" --output-dir "$MODEL_DIR" --max-length "$GRAPH_EVALUATOR_MAX_LENGTH" --epochs "$GRAPH_EVALUATOR_EPOCHS" --batch-size "$GRAPH_EVALUATOR_BATCH_SIZE" --gradient-accumulation-steps "$GRAPH_EVALUATOR_GRAD_ACCUM" --seed "$GRAPH_EVALUATOR_SEED" 2>&1 | tee "$TRAIN_LOG"

if [ ! -s "$MODEL_DIR/config.json" ] || [ ! -s "$MODEL_DIR/graph_rpo_calibration.json" ] || [ ! -s "$MODEL_DIR/tokenizer_config.json" ] || ! find -L "$MODEL_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: evaluator checkpoint is incomplete: $MODEL_DIR"
  exit 1
fi

python -u scripts/serve_graph_evaluator.py --model "$MODEL_DIR" --host 127.0.0.1 --port "$GRAPH_EVALUATOR_PROBE_PORT" --device cuda --max-length "$GRAPH_EVALUATOR_MAX_LENGTH" --local-files-only >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup_server() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

READY=0
for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${GRAPH_EVALUATOR_PROBE_PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  echo "ERROR: evaluator failed its HTTP health probe."
  tail -80 "$SERVER_LOG" || true
  exit 1
fi
curl --noproxy '*' -fsS -H 'Content-Type: application/json' -d '{"schema_version":"contextgraph.graph_evaluator.v1","items":[{"question":"Pilot health probe","graph_view":"[n1] root question"}]}' "http://127.0.0.1:${GRAPH_EVALUATOR_PROBE_PORT}/score"
echo

echo "=============================================================="
echo "  EVALUATOR PILOT COMPLETED"
echo "  GRAPH_RPO_EVALUATOR_MODEL=$MODEL_DIR"
echo "  Evaluator data: $DATA_DIR"
echo "  Data manifest: $DATA_DIR/manifest.json"
echo "  Training log:  $TRAIN_LOG"
echo "  NOTE: fine-tune and recalibrate on held-out BrowseComp rollouts before the formal run."
echo "=============================================================="
