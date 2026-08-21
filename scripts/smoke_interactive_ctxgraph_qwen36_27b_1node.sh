#!/bin/bash
#SBATCH -J cg-mt-27b
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/cg-mt-27b.%A_%a.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/cg-mt-27b.%A_%a.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 01:00:00
#SBATCH -A AST24021
#SBATCH --array=0-1

# Array task 0 runs ALFWorld; task 1 runs ScienceWorld. Each task starts a
# one-GH200 Qwen3.6-27B vLLM server and evaluates two official training tasks.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-27B}
SERVER_CONDA_ENV=${SERVER_CONDA_ENV:-deepseek_v4}
AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
SERVER_PORT=${SERVER_PORT:-18000}
MAX_SAMPLES=${MAX_SAMPLES:-2}
START_INDEX=${START_INDEX:-0}
MAX_TURN=${MAX_TURN:-30}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-1024}
PROMPT_LENGTH=${PROMPT_LENGTH:-16384}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
SCIENCEWORLD_VERSION=${SCIENCEWORLD_VERSION:-1.2.3}
RUN_TAG=${RUN_TAG:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}}

if [ -n "${DOMAIN:-}" ]; then
  case "$DOMAIN" in
    alfworld|scienceworld) ;;
    *) echo "ERROR: DOMAIN must be alfworld or scienceworld"; exit 2 ;;
  esac
else
  case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) DOMAIN=alfworld ;;
    1) DOMAIN=scienceworld ;;
    *) echo "ERROR: array index must be 0 or 1"; exit 2 ;;
  esac
fi

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/qwen3_6_27b_interactive_smoke/$RUN_TAG/$DOMAIN}
RAW_OUTPUT_DIR=$ARTIFACT_ROOT/raw
SFT_OUTPUT=$ARTIFACT_ROOT/contextgraph_sft_train.parquet
SCIENCEWORLD_DEPS=${SCIENCEWORLD_DEPS:-$SCRATCH/contextgraph_deps/scienceworld-$SCIENCEWORLD_VERSION}

resolve_snapshot() {
  local repo_id=$1
  shift
  local cache_name="models--${repo_id//\//--}"
  local hub
  local snapshot
  for hub in "$@"; do
    snapshot=$(find "$hub/$cache_name/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
    if [ -n "$snapshot" ] && [ -s "$snapshot/config.json" ]; then
      printf '%s\n' "$snapshot"
      return 0
    fi
  done
  return 1
}

require_scratch_path() {
  local label=$1
  local path=$2
  local scratch_real
  local path_real
  scratch_real=$(realpath -m "$SCRATCH")
  path_real=$(realpath -m "$path")
  case "$path_real" in
    "$scratch_real"/*) ;;
    *) echo "ERROR: $label must be stored under SCRATCH=$scratch_real, got $path_real"; exit 2 ;;
  esac
}

require_scratch_path HF_HOME "$HF_HOME"
require_scratch_path HF_HUB_CACHE "$HF_HUB_CACHE"

if [ -n "${MODEL_PATH:-}" ]; then
  test -s "$MODEL_PATH/config.json" || { echo "ERROR: invalid MODEL_PATH=$MODEL_PATH"; exit 2; }
else
  MODEL_PATH=$(resolve_snapshot "$MODEL_ID" "$HF_HUB_CACHE" "$SCRATCH/hf_cache") || true
fi
test -n "${MODEL_PATH:-}" || { echo "ERROR: $MODEL_ID is not cached"; exit 2; }
require_scratch_path MODEL_PATH "$MODEL_PATH"

mkdir -p "$PROJECT_ROOT/logs" "$RAW_OUTPUT_DIR"
cd "$PROJECT_ROOT"
export HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$AGENT_CONDA_ENV"
set -u

if [ "$DOMAIN" = alfworld ]; then
  export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
  test -d "$ALFWORLD_DATA/json_2.1.1/train" || { echo "ERROR: ALFWorld train data missing under $ALFWORLD_DATA"; exit 2; }
  python scripts/make_alfworld_data.py --mode real --n_train "$((START_INDEX + MAX_SAMPLES))" --n_val 1 --seed 42
  DATA_PATH=data/alfworld_graph_real_train.parquet
  WORKFLOW=alfworld_graph
else
  if [ ! -s "$SCIENCEWORLD_DEPS/scienceworld/scienceworld.jar" ]; then
    mkdir -p "$SCIENCEWORLD_DEPS"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -m pip install --target "$SCIENCEWORLD_DEPS" --no-deps "scienceworld==$SCIENCEWORLD_VERSION" py4j
  fi
  export PYTHONPATH="$SCIENCEWORLD_DEPS:$PYTHONPATH"
  java -version
  python scripts/make_scienceworld_data.py --n-train "$((START_INDEX + MAX_SAMPLES))" --n-val 1 --seed 42
  DATA_PATH=data/scienceworld_graph_train.parquet
  WORKFLOW=scienceworld_graph
fi

set +u
conda activate "$SERVER_CONDA_ENV"
set -u
python scripts/check_hf_model_support.py "$MODEL_PATH"
VLLM_HELP=$(vllm serve --help=all 2>&1)
for required_flag in --reasoning-parser --language-model-only; do
  printf '%s\n' "$VLLM_HELP" | grep -q -- "$required_flag" || { echo "ERROR: vLLM lacks $required_flag"; exit 2; }
done

VLLM_LOG=$PROJECT_ROOT/logs/cg-mt-27b-vllm.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}.log
vllm serve "$MODEL_PATH" --served-model-name "$MODEL_ID" --host 127.0.0.1 --port "$SERVER_PORT" --tensor-parallel-size 1 --trust-remote-code --reasoning-parser qwen3 --language-model-only --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --max-num-seqs 2 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --enable-chunked-prefill >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!
cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 1200); do
  curl --noproxy '*' -fsS "http://127.0.0.1:$SERVER_PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "ERROR: vLLM exited; inspect $VLLM_LOG"; exit 3; }
  sleep 1
done
curl --noproxy '*' -fsS "http://127.0.0.1:$SERVER_PORT/v1/models" >/dev/null || { echo "ERROR: vLLM health timeout"; exit 3; }

set +u
conda activate "$AGENT_CONDA_ENV"
set -u
if [ "$DOMAIN" = scienceworld ]; then
  export PYTHONPATH="$SCIENCEWORLD_DEPS:$PROJECT_ROOT:${PYTHONPATH:-}"
fi
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL="http://127.0.0.1:$SERVER_PORT/v1"
export QWEN_ENABLE_THINKING=True

python scripts/eval_interactive.py --data-path "$DATA_PATH" --output-dir "$RAW_OUTPUT_DIR" --workflow "$WORKFLOW" --model-name "$MODEL_ID" --tokenizer-name "$MODEL_PATH" --max-samples "$MAX_SAMPLES" --start-index "$START_INDEX" --num-workers 1 --prompt-length "$PROMPT_LENGTH" --response-length "$RESPONSE_LENGTH" --max-turn "$MAX_TURN" --max-session 4 --branch-len 8192 --turn-max-new-tokens "$TURN_MAX_NEW_TOKENS" --temperature 0.6 --top-p 0.95 --save-messages

RESULT_FILE=$(find "$RAW_OUTPUT_DIR" -maxdepth 1 -name 'interactive_results_*.json' -type f | sort | tail -n 1)
test -n "$RESULT_FILE" || { echo "ERROR: evaluator produced no result JSON"; exit 3; }
python scripts/validate_contextgraph_traces.py "$RESULT_FILE"

if python scripts/build_contextgraph_sft.py "$RESULT_FILE" --output "$SFT_OUTPUT" --validation-fraction 0.0 --min-task-reward 1.0 --min-valid-graph-ops 1 --min-structural-graph-ops 1 --max-invalid-graph-ops 0 --require-graph-trace --min-graph-quality-score 1.0 --max-redundant-graph-ops 0 --teacher-provider local_vllm --teacher-model "$MODEL_ID"; then
  echo "Strict SFT smoke output: $SFT_OUTPUT"
else
  echo "WARN: no trajectory passed strict SFT filters; raw smoke remains valid at $RESULT_FILE"
fi
echo "Interactive Qwen3.6-27B smoke complete: domain=$DOMAIN artifacts=$ARTIFACT_ROOT"
