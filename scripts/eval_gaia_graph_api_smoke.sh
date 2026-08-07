#!/bin/bash
#SBATCH -J eval-gaia-graph-api-smoke
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/eval-gaia-graph-api-smoke.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/eval-gaia-graph-api-smoke.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 00:45:00
#SBATCH -A AST24021

# Single-node GAIA API smoke. This mirrors the BrowseComp search-server
# preflight pattern, but does not start Ray because scripts/eval_gaia.py calls
# an OpenAI-compatible model API directly.
set -euo pipefail

LOG_ROOT=/work/09281/chc_1996/vista/context-graph/logs
mkdir -p "$LOG_ROOT"

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1

if [ -n "${WORK:-}" ] && [ -f "$WORK/.openai_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.openai_env"
fi
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "dummy" ]; then
  echo "ERROR: OPENAI_API_KEY not set. GAIA eval requires the OpenAI-compatible API client."
  echo "       echo 'export OPENAI_API_KEY=sk-...' > \$WORK/.openai_env && chmod 600 \$WORK/.openai_env"
  exit 1
fi

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
set -u
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}
MODEL_NAME=${GAIA_MODEL_NAME:-gpt-5-nano}
TOKENIZER_NAME=${GAIA_TOKENIZER_NAME:-Qwen/Qwen2.5-7B-Instruct}
WORKFLOW=${GAIA_WORKFLOW:-search_graph}
MAX_SAMPLES=${GAIA_MAX_SAMPLES:-1}
NUM_WORKERS=${GAIA_NUM_WORKERS:-1}
PROMPT_LENGTH=${GAIA_PROMPT_LENGTH:-16384}
RESPONSE_LENGTH=${GAIA_RESPONSE_LENGTH:-16384}
MAX_TURN=${GAIA_MAX_TURN:-12}
MAX_SESSION=${GAIA_MAX_SESSION:-4}
BRANCH_LEN=${GAIA_BRANCH_LEN:-8192}
TURN_MAX_NEW_TOKENS=${GAIA_TURN_MAX_NEW_TOKENS:-512}
SEARCH_TOPK_CAP=${GAIA_SEARCH_TOPK_CAP:-5}
SEARCH_SNIPPET_WORDS=${GAIA_SEARCH_SNIPPET_WORDS:-128}
SEARCH_SNIPPET_CHARS=${GAIA_SEARCH_SNIPPET_CHARS:-2000}
OPEN_PAGE_WORDS=${GAIA_OPEN_PAGE_WORDS:-1024}
OPEN_PAGE_CHARS=${GAIA_OPEN_PAGE_CHARS:-12000}
SEARCH_PORT=${GAIA_SEARCH_PORT:-18999}
SEARCH_TIMEOUT_SECONDS=${GAIA_SEARCH_TIMEOUT_SECONDS:-240}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

case "$WORKFLOW" in
  search) DEFAULT_DATA_PATH=data/gaia_validation.parquet ;;
  search_branch) DEFAULT_DATA_PATH=data/gaia_validation_branch.parquet ;;
  search_graph) DEFAULT_DATA_PATH=data/gaia_validation_graph.parquet ;;
  *)
    echo "ERROR: unsupported GAIA_WORKFLOW=$WORKFLOW; expected search, search_branch, or search_graph"
    exit 1
    ;;
esac
DATA_PATH=${GAIA_DATA_PATH:-$DEFAULT_DATA_PATH}

CORPUS_DATASET="Tevatron/browsecomp-plus-corpus"
CORPUS_EMBEDDING_DATASET="miaolu3/browsecomp-plus"
CORPUS_CACHE_DIR="$HF_HOME/hub/datasets--${CORPUS_DATASET//\//--}"
EMBED_DATA_CACHE_DIR="$HF_HOME/hub/datasets--${CORPUS_EMBEDDING_DATASET//\//--}"
EMBED_MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${EMBED_MODEL//\//--}"

echo "=============================================================="
echo "  GAIA API smoke: workflow=$WORKFLOW samples=$MAX_SAMPLES workers=$NUM_WORKERS"
echo "  Project:        $PROJECT_ROOT"
echo "  Data:           $DATA_PATH"
echo "  Model API:      $MODEL_NAME"
echo "  Embedder:       $EMBED_MODEL"
echo "  Search port:    $SEARCH_PORT"
echo "  Token caps:     prompt=$PROMPT_LENGTH response=$RESPONSE_LENGTH turn=$TURN_MAX_NEW_TOKENS"
echo "  Agent caps:     max_turn=$MAX_TURN max_session=$MAX_SESSION branch_len=$BRANCH_LEN"
echo "  Search bounds:  topk=$SEARCH_TOPK_CAP snippet=${SEARCH_SNIPPET_WORDS}w/${SEARCH_SNIPPET_CHARS}c open=${OPEN_PAGE_WORDS}w/${OPEN_PAGE_CHARS}c"
echo "  Started:        $(date)"
echo "=============================================================="

probe "checking GAIA parquet and HF caches"
if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: missing $PROJECT_ROOT/$DATA_PATH"
  echo "       Run: python scripts/make_gaia_data.py --split validation --out-dir data"
  exit 1
fi
if [ ! -d "$EMBED_MODEL_CACHE_DIR" ]; then
  echo "ERROR: embedder model not cached at $EMBED_MODEL_CACHE_DIR"
  echo "       Login node: hf download $EMBED_MODEL"
  exit 1
fi
if [ ! -d "$CORPUS_CACHE_DIR" ]; then
  echo "ERROR: corpus dataset not cached at $CORPUS_CACHE_DIR"
  echo "       Login node: hf download $CORPUS_DATASET --repo-type=dataset"
  exit 1
fi
if [ ! -d "$EMBED_DATA_CACHE_DIR" ]; then
  echo "ERROR: embedding dataset not cached at $EMBED_DATA_CACHE_DIR"
  echo "       Login node: hf download $CORPUS_EMBEDDING_DATASET --repo-type=dataset"
  exit 1
fi
probe "GAIA parquet + HF caches ok"

SEARCH_LOG=/tmp/gaia_search_${SLURM_JOB_ID:-local}_$$.log
probe "starting envs/search_server.py on localhost:$SEARCH_PORT"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export NUM_GPUS=1
export MAX_BATCH_SIZE=128
unset LOCAL_CORPUS_PARQUET LOCAL_EMBEDDINGS_PKL
python -u envs/search_server.py \
  --model "$EMBED_MODEL" \
  --host 0.0.0.0 \
  --port "$SEARCH_PORT" \
  --corpus "$CORPUS_DATASET" \
  --corpus-embedding-dataset "$CORPUS_EMBEDDING_DATASET" \
  > "$SEARCH_LOG" 2>&1 &
SEARCH_PID=$!

cleanup() {
  kill "$SEARCH_PID" 2>/dev/null || true
}
trap cleanup EXIT

probe "waiting for search server /health (up to ${SEARCH_TIMEOUT_SECONDS}s)"
HEALTH_OK=0
for _ in $(seq 1 "$SEARCH_TIMEOUT_SECONDS"); do
  if curl -fsS "http://127.0.0.1:${SEARCH_PORT}/health" >/dev/null 2>&1; then
    HEALTH_OK=1
    break
  fi
  if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
    echo "ERROR: search server exited before becoming healthy. Last 80 lines:"
    tail -80 "$SEARCH_LOG" || true
    exit 1
  fi
  sleep 1
done
if [ "$HEALTH_OK" != "1" ]; then
  echo "ERROR: search server did not become healthy within ${SEARCH_TIMEOUT_SECONDS}s. Last 80 lines:"
  tail -80 "$SEARCH_LOG" || true
  exit 1
fi

probe "waiting for search server /search probe (up to ${SEARCH_TIMEOUT_SECONDS}s)"
SEARCH_OK=0
for _ in $(seq 1 "$SEARCH_TIMEOUT_SECONDS"); do
  if curl -fsS -X POST -H 'Content-Type: application/json' \
      -d '{"query":"Eiffel Tower","k":1}' \
      "http://127.0.0.1:${SEARCH_PORT}/search" >/dev/null 2>&1; then
    SEARCH_OK=1
    break
  fi
  if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
    echo "ERROR: search server exited before /search probe succeeded. Last 80 lines:"
    tail -80 "$SEARCH_LOG" || true
    exit 1
  fi
  sleep 1
done
if [ "$SEARCH_OK" != "1" ]; then
  echo "ERROR: search server probe failed. Last 80 lines:"
  tail -80 "$SEARCH_LOG" || true
  exit 1
fi
export LOCAL_SEARCH_URL="http://127.0.0.1:${SEARCH_PORT}"
probe "search server up at $LOCAL_SEARCH_URL"

probe "running GAIA eval"
SAVE_MESSAGE_ARGS=()
if [ "${GAIA_SAVE_MESSAGES:-0}" = "1" ]; then
  SAVE_MESSAGE_ARGS+=(--save-messages)
fi
python scripts/eval_gaia.py \
  --data-path "$DATA_PATH" \
  --workflow "$WORKFLOW" \
  --max-samples "$MAX_SAMPLES" \
  --num-workers "$NUM_WORKERS" \
  --local-search-url "$LOCAL_SEARCH_URL" \
  --model-name "$MODEL_NAME" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --prompt-length "$PROMPT_LENGTH" \
  --response-length "$RESPONSE_LENGTH" \
  --max-turn "$MAX_TURN" \
  --max-session "$MAX_SESSION" \
  --branch-len "$BRANCH_LEN" \
  --turn-max-new-tokens "$TURN_MAX_NEW_TOKENS" \
  --search-topk-cap "$SEARCH_TOPK_CAP" \
  --search-snippet-words "$SEARCH_SNIPPET_WORDS" \
  --search-snippet-chars "$SEARCH_SNIPPET_CHARS" \
  --open-page-words "$OPEN_PAGE_WORDS" \
  --open-page-chars "$OPEN_PAGE_CHARS" \
  "${SAVE_MESSAGE_ARGS[@]}"
probe "GAIA eval done"
