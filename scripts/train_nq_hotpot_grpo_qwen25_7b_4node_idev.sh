#!/bin/bash

# Run inside an active four-node Vista idev allocation. Node 0 runs the
# official Search-R1 Wiki-18 E5/FAISS retriever; nodes 1-3 run Qwen2.5-7B GRPO.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SEARCH_DATA_ROOT=${SEARCH_DATA_ROOT:-${SCRATCH:?SCRATCH must be set}/context-graph-data/searchr1_nq_hotpotqa}
DATA_ROOT=${DATA_ROOT:-$SEARCH_DATA_ROOT/processed}
RETRIEVER_ROOT=${RETRIEVER_ROOT:-$SEARCH_DATA_ROOT/wiki18}
RETRIEVER_ENV=${RETRIEVER_ENV:-$SEARCH_DATA_ROOT/wiki18_retriever_env}
E5_MODEL_DIR=${E5_MODEL_DIR:-$RETRIEVER_ROOT/e5-base-v2}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
WIKI18_SEARCH_TIMEOUT_SECONDS=${WIKI18_SEARCH_TIMEOUT_SECONDS:-1800}
TS=$(date +%Y%m%d_%H%M%S)
RUN_LOG=${RUN_LOG:-$PROJECT_ROOT/logs/train-searchr1-nq-hotpot-qwen25-7b-$TS.log}
SEARCH_LOG=${SEARCH_LOG:-$PROJECT_ROOT/logs/wiki18-search-$TS.log}

run_wiki18_server() {
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate "$RETRIEVER_ENV"
  export HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd "$PROJECT_ROOT"
  exec python -u envs/wiki18_search_server.py \
    --index-path "$RETRIEVER_ROOT/e5_Flat.index" \
    --corpus-path "$RETRIEVER_ROOT/wiki-18.jsonl" \
    --model-path "$E5_MODEL_DIR" \
    --port 18999 --faiss-gpu
}

if [ "${1:-}" = wiki18_server ]; then
  run_wiki18_server
fi

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this inside an active four-node idev allocation."
  exit 2
fi

mkdir -p "$PROJECT_ROOT/logs"

for path in \
  "$DATA_ROOT/train.parquet" \
  "$DATA_ROOT/validation_diag.parquet" \
  "$RETRIEVER_ROOT/e5_Flat.index" \
  "$RETRIEVER_ROOT/wiki-18.jsonl" \
  "$E5_MODEL_DIR/config.json" \
  "$RETRIEVER_ENV/bin/python"; do
  if [ ! -s "$path" ]; then
    echo "ERROR: missing Wiki-18 experiment artifact: $path"
    echo "Run scripts/download_nq_hotpot_search_data_vista.sh on a login node first."
    exit 1
  fi
done

mapfile -t IDEV_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#IDEV_NODES[@]}" -ne 4 ]; then
  echo "ERROR: expected four idev nodes, got ${#IDEV_NODES[@]}"
  exit 2
fi
SEARCH_NODE=${IDEV_NODES[0]}
SEARCH_NODE_IP=$(getent hosts "$SEARCH_NODE" | awk '{print $1}')
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$SEARCH_NODE,$SEARCH_NODE_IP"
export no_proxy="$NO_PROXY"

echo "Starting Wiki-18 search on $SEARCH_NODE ($SEARCH_NODE_IP); log=$SEARCH_LOG"
srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=1 -w "$SEARCH_NODE" \
  bash "$PROJECT_ROOT/scripts/train_nq_hotpot_grpo_qwen25_7b_4node_idev.sh" wiki18_server \
  >"$SEARCH_LOG" 2>&1 &
WIKI18_SEARCH_PID=$!
cleanup() {
  kill "$WIKI18_SEARCH_PID" 2>/dev/null || true
}
trap cleanup EXIT

SEARCH_URL="http://${SEARCH_NODE_IP}:18999"
echo "Waiting up to ${WIKI18_SEARCH_TIMEOUT_SECONDS}s for Wiki-18 corpus/index startup"
SEARCH_READY=0
for _ in $(seq 1 "$WIKI18_SEARCH_TIMEOUT_SECONDS"); do
  if curl --noproxy '*' -fsS "$SEARCH_URL/health" >/dev/null 2>&1; then
    SEARCH_READY=1
    break
  fi
  if ! kill -0 "$WIKI18_SEARCH_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ "$SEARCH_READY" != 1 ]; then
  echo "ERROR: Wiki-18 search failed to become healthy. Last 100 lines:"
  tail -100 "$SEARCH_LOG" || true
  exit 1
fi
curl --noproxy '*' -fsS -X POST -H 'Content-Type: application/json' \
  -d '{"query":"Who wrote Pride and Prejudice?","k":3}' "$SEARCH_URL/search"
echo
export EXTERNAL_SEARCH_URL="$SEARCH_URL"

export TASK_LABEL="Search-R1 NQ+HotpotQA GRPO diagnostic"
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
export TRAIN_DATA_FILE=${TRAIN_DATA_FILE:-$DATA_ROOT/train.parquet}
export VAL_DATA_FILE=${VAL_DATA_FILE:-$DATA_ROOT/validation_diag.parquet}
export REQUIRE_OPENAI_JUDGE=0
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-12}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TRAIN_LR=${TRAIN_LR:-1e-6}
export PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-10240}
export MAX_TURN=${MAX_TURN:-16}
export MAX_SESSION=${MAX_SESSION:-4}
export VAL_MAX_SESSION=${VAL_MAX_SESSION:-4}
export TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-512}
export FINAL_ANSWER_RESERVE=${FINAL_ANSWER_RESERVE:-1024}
export RUN_TAG=${RUN_TAG:-searchr1_nq_hotpot_qwen25_7b_grpo_10step}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$RUN_TAG-$TS}
export BC_SEARCH_TIMEOUT_SECONDS=60

set +e
bash "$PROJECT_ROOT/scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh" 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e

echo "Training log: $RUN_LOG"
echo "Wiki-18 search log: $SEARCH_LOG"
if [ "$RC" -ne 0 ]; then
  exit "$RC"
fi
python "$PROJECT_ROOT/scripts/audit_skillrl_search_reference.py" --sources searchR1_nq,searchR1_hotpotqa --require-training-health "$RUN_LOG"
