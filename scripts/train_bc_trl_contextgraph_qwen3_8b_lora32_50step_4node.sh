#!/bin/bash
#SBATCH -J bc-trl-cg-8b-lora50
#SBATCH -o logs/bc-trl-cg-8b-lora50.%j.out
#SBATCH -e logs/bc-trl-cg-8b-lora50.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH -A AST24021

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
CONDA_BASE=${CONDA_BASE:-/work/09281/chc_1996/vista/miniconda3}
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}
SEARCH_PORT=${SEARCH_PORT:-18999}
BC_SEARCH_TIMEOUT_SECONDS=${BC_SEARCH_TIMEOUT_SECONDS:-600}

if [ -n "${WORK:-}" ] && [ -f "$WORK/.openai_env" ]; then
  source "$WORK/.openai_env"
fi
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = dummy ]; then
  echo "ERROR: BC-P training requires OPENAI_API_KEY for the answer judge." >&2
  exit 1
fi
if [ -n "${WORK:-}" ] && [ -f "$WORK/.wandb_env" ]; then
  source "$WORK/.wandb_env"
fi

cd "$PROJECT_ROOT"
mkdir -p logs
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-$PROJECT_ROOT/data/bc_train.parquet}
if [ ! -s "$TRAIN_DATA_PATH" ]; then
  echo "ERROR: BC-P training parquet missing or empty: $TRAIN_DATA_PATH" >&2
  exit 1
fi

mapfile -t BC_TRL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#BC_TRL_NODES[@]}" -lt 4 ]; then
  echo "ERROR: expected at least 4 nodes, got ${#BC_TRL_NODES[@]}" >&2
  exit 1
fi
SEARCH_NODES=("${BC_TRL_NODES[@]:0:3}")
TRAIN_NODE=${BC_TRL_NODES[3]}
if [ "${#BC_TRL_NODES[@]}" -gt 4 ]; then
  echo "Using the first 4 nodes; leaving $(( ${#BC_TRL_NODES[@]} - 4 )) extra allocation node(s) idle."
fi

export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NUM_GPUS=1 MAX_BATCH_SIZE=128 PYTHONUNBUFFERED=1

SEARCH_PIDS=()
SEARCH_URLS=()
SEARCH_LOGS=()
cleanup() {
  for pid in "${SEARCH_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${SEARCH_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

echo "4-node topology: retrievers=${SEARCH_NODES[*]}; TRL trainer=$TRAIN_NODE"
for SEARCH_INDEX in 0 1 2; do
  SEARCH_NODE=${SEARCH_NODES[$SEARCH_INDEX]}
  SEARCH_NODE_IP=$(getent hosts "$SEARCH_NODE" | awk '{print $1; exit}')
  SEARCH_URL=http://$SEARCH_NODE_IP:$SEARCH_PORT
  SEARCH_LOG=$PROJECT_ROOT/logs/bc-search-${SLURM_JOB_ID:-idev}-${SEARCH_NODE}.log
  SEARCH_URLS+=("$SEARCH_URL")
  SEARCH_LOGS+=("$SEARCH_LOG")
  echo "Starting BC-P retriever $((SEARCH_INDEX + 1))/3 on $SEARCH_NODE"
  srun --overlap --nodes=1 --ntasks=1 -w "$SEARCH_NODE" bash -lc "source $CONDA_BASE/etc/profile.d/conda.sh; conda activate cxtgraph; cd $PROJECT_ROOT; export PYTHONPATH=$PROJECT_ROOT:\${PYTHONPATH:-}; exec python -u envs/search_server.py --model $EMBED_MODEL --host 0.0.0.0 --port $SEARCH_PORT --corpus Tevatron/browsecomp-plus-corpus --corpus-embedding-dataset miaolu3/browsecomp-plus" >"$SEARCH_LOG" 2>&1 &
  SEARCH_PIDS+=("$!")
done

SEARCH_READY=(0 0 0)
SEARCH_DEADLINE=$((SECONDS + BC_SEARCH_TIMEOUT_SECONDS))
SEARCH_ATTEMPT=0
while [ "$SECONDS" -lt "$SEARCH_DEADLINE" ]; do
  SEARCH_ATTEMPT=$((SEARCH_ATTEMPT + 1))
  ALL_SEARCH_READY=1
  for SEARCH_INDEX in 0 1 2; do
    if [ "${SEARCH_READY[$SEARCH_INDEX]}" = 1 ]; then
      continue
    fi
    ALL_SEARCH_READY=0
    if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS -X POST -H 'Content-Type: application/json' -d '{"query":"Eiffel Tower","k":1}' "${SEARCH_URLS[$SEARCH_INDEX]}/search" >/dev/null 2>&1; then
      SEARCH_READY[$SEARCH_INDEX]=1
      echo "Retriever ready: ${SEARCH_NODES[$SEARCH_INDEX]}"
      continue
    fi
    if ! kill -0 "${SEARCH_PIDS[$SEARCH_INDEX]}" 2>/dev/null; then
      echo "ERROR: retriever exited on ${SEARCH_NODES[$SEARCH_INDEX]}." >&2
      tail -80 "${SEARCH_LOGS[$SEARCH_INDEX]}" || true
      exit 1
    fi
  done
  if [ "$ALL_SEARCH_READY" = 1 ]; then
    break
  fi
  if [ $((SEARCH_ATTEMPT % 2)) -eq 0 ]; then
    echo "Waiting for retrievers ($((BC_SEARCH_TIMEOUT_SECONDS - (SEARCH_DEADLINE - SECONDS)))s elapsed); ready=${SEARCH_READY[*]}"
    for SEARCH_INDEX in 0 1 2; do
      echo "--- ${SEARCH_NODES[$SEARCH_INDEX]} latest log ---"
      tail -5 "${SEARCH_LOGS[$SEARCH_INDEX]}" || true
    done
  fi
  sleep 1
done
if [ "${SEARCH_READY[*]}" != "1 1 1" ]; then
  echo "ERROR: not all BC-P retrievers were ready within ${BC_SEARCH_TIMEOUT_SECONDS}s; ready=${SEARCH_READY[*]}." >&2
  exit 1
fi

LOCAL_SEARCH_URL=$(IFS=,; echo "${SEARCH_URLS[*]}")
export LOCAL_SEARCH_URL
export AGENT_KIND=contextgraph
export AGENT_CONFIG=$PROJECT_ROOT/recipes/trl_agent/contextgraph_browsecomp_plus.yaml
export DATASET_NAME=browsecomp-plus
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
export SAVE_STEPS=${SAVE_STEPS:-10}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-128}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-8192}
export MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-10240}
export PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
export NUM_GENERATIONS=${NUM_GENERATIONS:-16}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.40}
export RUN_TAG=${RUN_TAG:-trl-contextgraph-bc-qwen3-8b-lora32-50step-4node}
export WANDB_TAGS=${WANDB_TAGS:-trl,contextgraph,browsecomp-plus,qwen3-8b,lora32,g16,50step,4node}

echo "Launching TRL ContextGraph BC-P: retriever_pool=$LOCAL_SEARCH_URL trainer=$TRAIN_NODE model=$MODEL_PATH LoRA=32 steps=$TOTAL_TRAINING_STEPS"
srun --overlap --nodes=1 --ntasks=1 -w "$TRAIN_NODE" bash "$PROJECT_ROOT/scripts/train_gsm8k_trl_agent_lora32.sh"
