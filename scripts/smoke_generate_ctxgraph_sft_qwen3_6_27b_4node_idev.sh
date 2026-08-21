#!/bin/bash

# Run a small executor-verified ContextGraph SFT smoke inside an existing
# Vista idev allocation. One GH200 serves Qwen3.6-27B and one serves the
# BrowseComp-Plus retriever; any remaining allocation nodes stay unused.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.6-27B}
SERVER_CONDA_ENV=${SERVER_CONDA_ENV:-deepseek_v4}
AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
TEACHER_PORT=${TEACHER_PORT:-18000}
SEARCH_PORT=${SEARCH_PORT:-18999}
MAX_SAMPLES=${MAX_SAMPLES:-2}
START_INDEX=${START_INDEX:-0}
NUM_WORKERS=${NUM_WORKERS:-1}
MAX_TURN=${MAX_TURN:-12}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-4096}
PROMPT_LENGTH=${PROMPT_LENGTH:-16384}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
DATA_PATH=${DATA_PATH:-data/bc_train.parquet}
ALLOW_EVAL_DATA=${ALLOW_EVAL_DATA:-0}
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
SHARED_HF_HOME=${SHARED_HF_HOME:-/work/09281/chc_1996/vista/cache}
SHARED_HF_HUB_CACHE=${SHARED_HF_HUB_CACHE:-$SHARED_HF_HOME/hub}
SEARCH_HF_HOME=${SEARCH_HF_HOME:-$SHARED_HF_HOME}
SEARCH_HF_HUB_CACHE=${SEARCH_HF_HUB_CACHE:-$SEARCH_HF_HOME/hub}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/qwen3_6_27b_smoke/$RUN_TAG}
RAW_OUTPUT_DIR=$ARTIFACT_ROOT/raw
SFT_OUTPUT=$ARTIFACT_ROOT/contextgraph_sft_train.parquet
SFT_VALIDATION_OUTPUT=$ARTIFACT_ROOT/contextgraph_sft_validation.parquet

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

if [ -n "${MODEL_PATH:-}" ]; then
  if [ ! -s "$MODEL_PATH/config.json" ]; then
    echo "ERROR: MODEL_PATH does not contain config.json: $MODEL_PATH"
    exit 2
  fi
else
  MODEL_PATH=$(resolve_snapshot "$MODEL_ID" "$HF_HUB_CACHE" "$SHARED_HF_HUB_CACHE" "$SCRATCH/hf_cache") || true
fi
if [ -z "${MODEL_PATH:-}" ]; then
  echo "ERROR: $MODEL_ID is not cached."
  echo "Download it first: HF_HOME=$HF_HOME hf download $MODEL_ID"
  echo "Or set MODEL_PATH=/absolute/path/to/Qwen3.6-27B"
  exit 2
fi

mkdir -p "$PROJECT_ROOT/logs" "$RAW_OUTPUT_DIR"
cd "$PROJECT_ROOT"
if [ ! -s "$DATA_PATH" ]; then
  echo "ERROR: missing or empty seed parquet: $PROJECT_ROOT/$DATA_PATH"
  exit 2
fi
case "$DATA_PATH" in
  *validation*|*test*)
    if [ "$ALLOW_EVAL_DATA" != "1" ]; then
      echo "ERROR: refusing evaluation split $DATA_PATH to prevent benchmark contamination"
      exit 2
    fi
    ;;
esac

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$SERVER_CONDA_ENV"
set -u
python -c "import transformers, vllm; from packaging.version import Version; from transformers import AutoConfig; assert Version(vllm.__version__) >= Version('0.19.0'), 'Qwen3.6 requires vLLM >= 0.19.0'; c=AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True, local_files_only=True); print('server preflight:', 'transformers='+transformers.__version__, 'vllm='+vllm.__version__, 'model_type='+str(getattr(c, 'model_type', None)))"
# vLLM 0.27 uses paged/grouped CLI help; plain --help intentionally omits
# model and parallelism options.
VLLM_HELP=$(vllm serve --help=all 2>&1)
for required_flag in --tensor-parallel-size --reasoning-parser --language-model-only; do
  if ! printf '%s\n' "$VLLM_HELP" | grep -q -- "$required_flag"; then
    echo "ERROR: $SERVER_CONDA_ENV vLLM does not support $required_flag"
    exit 2
  fi
done
if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "Qwen3.6-27B smoke preflight passed."
  echo "Model path: $MODEL_PATH"
  exit 0
fi

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this script inside an active Vista idev/Slurm allocation"
  exit 2
fi
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NUM_NODES=${#NODELIST[@]}
if [ "$NUM_NODES" -lt 2 ]; then
  echo "ERROR: at least 2 allocated nodes are required; got $NUM_NODES"
  exit 2
fi

SEARCH_NODE=${NODELIST[0]}
SEARCH_NODE_IP=$(getent hosts "$SEARCH_NODE" | awk '{print $1}')
TEACHER_NODE=${NODELIST[1]}
TEACHER_NODE_IP=$(getent hosts "$TEACHER_NODE" | awk '{print $1}')
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$SEARCH_NODE,$SEARCH_NODE_IP,$TEACHER_NODE,$TEACHER_NODE_IP"
export no_proxy=$NO_PROXY

echo "Model:      $MODEL_ID"
echo "Model path: $MODEL_PATH"
echo "Teacher:    $TEACHER_NODE ($TEACHER_NODE_IP:$TEACHER_PORT)"
echo "Search:     $SEARCH_NODE ($SEARCH_NODE_IP:$SEARCH_PORT)"
echo "Samples:    $MAX_SAMPLES from index $START_INDEX"
echo "Output:     $ARTIFACT_ROOT"
if [ "$NUM_NODES" -gt 2 ]; then
  echo "Idle nodes:  ${NODELIST[*]:2}"
fi

STEP_PIDS=()
cleanup() {
  set +e
  for pid in "${STEP_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

SEARCH_LOG="$PROJECT_ROOT/logs/gen-cg-sft-qwen36-search.${SLURM_JOB_ID:-idev}.log"
srun --overlap --nodes=1 --ntasks=1 -w "$SEARCH_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $AGENT_CONDA_ENV; cd $PROJECT_ROOT; export HF_HOME=$SEARCH_HF_HOME HF_HUB_CACHE=$SEARCH_HF_HUB_CACHE NUM_GPUS=1 MAX_BATCH_SIZE=64; exec python -u envs/search_server.py --model $EMBED_MODEL --host 0.0.0.0 --port $SEARCH_PORT --corpus Tevatron/browsecomp-plus-corpus --corpus-embedding-dataset miaolu3/browsecomp-plus" >"$SEARCH_LOG" 2>&1 &
STEP_PIDS+=("$!")

for _ in $(seq 1 600); do
  if curl --noproxy '*' -fsS "http://$SEARCH_NODE_IP:$SEARCH_PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl --noproxy '*' -fsS "http://$SEARCH_NODE_IP:$SEARCH_PORT/health" >/dev/null; then
  echo "ERROR: search server did not become healthy; inspect $SEARCH_LOG"
  exit 3
fi

VLLM_LOG="$PROJECT_ROOT/logs/gen-cg-sft-qwen36-vllm.${SLURM_JOB_ID:-idev}.log"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; exec vllm serve $MODEL_PATH --served-model-name $MODEL_ID --host 0.0.0.0 --port $TEACHER_PORT --tensor-parallel-size 1 --trust-remote-code --reasoning-parser qwen3 --language-model-only --dtype bfloat16 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --enable-chunked-prefill" >"$VLLM_LOG" 2>&1 &
STEP_PIDS+=("$!")

for _ in $(seq 1 1200); do
  if curl --noproxy '*' -fsS "http://$TEACHER_NODE_IP:$TEACHER_PORT/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! curl --noproxy '*' -fsS "http://$TEACHER_NODE_IP:$TEACHER_PORT/v1/models" >/dev/null; then
  echo "ERROR: vLLM server did not become healthy; inspect $VLLM_LOG"
  exit 3
fi

OPENAI_BASE_URL="http://$TEACHER_NODE_IP:$TEACHER_PORT/v1"
LOCAL_SEARCH_URL="http://$SEARCH_NODE_IP:$SEARCH_PORT"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $AGENT_CONDA_ENV; cd $PROJECT_ROOT; export OPENAI_API_KEY=dummy OPENAI_BASE_URL=$OPENAI_BASE_URL LOCAL_SEARCH_URL=$LOCAL_SEARCH_URL QWEN_ENABLE_THINKING=True; python scripts/eval_gaia.py --data-path $DATA_PATH --output-dir $RAW_OUTPUT_DIR --workflow search_graph --model-name $MODEL_ID --tokenizer-name $MODEL_PATH --num-workers $NUM_WORKERS --start-index $START_INDEX --max-samples $MAX_SAMPLES --prompt-length $PROMPT_LENGTH --response-length $RESPONSE_LENGTH --max-turn $MAX_TURN --max-session 8 --branch-len 8192 --turn-max-new-tokens $TURN_MAX_NEW_TOKENS --temperature $TEMPERATURE --top-p $TOP_P --local-search-url $LOCAL_SEARCH_URL --must-search --save-messages"

RESULT_FILE=$(find "$RAW_OUTPUT_DIR" -maxdepth 1 -name 'gaia_results_*.json' -type f | sort | tail -n 1)
if [ -z "$RESULT_FILE" ]; then
  echo "ERROR: evaluator produced no result JSON"
  exit 3
fi

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$AGENT_CONDA_ENV"
set -u
python scripts/build_contextgraph_sft.py "$RESULT_FILE" --output "$SFT_OUTPUT" --validation-output "$SFT_VALIDATION_OUTPUT" --validation-fraction 0.0 --min-task-reward 1.0 --min-valid-graph-ops 1 --min-structural-graph-ops 1 --max-invalid-graph-ops 0 --teacher-provider local_vllm --teacher-model "$MODEL_ID"
echo "Qwen3.6-27B ContextGraph SFT smoke complete: $SFT_OUTPUT"
