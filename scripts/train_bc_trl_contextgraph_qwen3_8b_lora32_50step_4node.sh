#!/bin/bash
# Four-node BC-P topology: one retrieval server plus three DDP TRL ranks.
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
BC_SEARCH_LAUNCH_TIMEOUT_SECONDS=${BC_SEARCH_LAUNCH_TIMEOUT_SECONDS:-30}

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
SEARCH_NODE=${BC_TRL_NODES[0]}
TRAIN_NODES=("${BC_TRL_NODES[@]:1:3}")
TRAIN_NODELIST=$(IFS=,; echo "${TRAIN_NODES[*]}")
SEARCH_NODE_IP=$(getent hosts "$SEARCH_NODE" | awk '{print $1; exit}')
TRAIN_MASTER_ADDR=$(getent hosts "${TRAIN_NODES[0]}" | awk '{print $1; exit}')
if [ "${#BC_TRL_NODES[@]}" -gt 4 ]; then
  echo "Using the first 4 nodes; leaving $(( ${#BC_TRL_NODES[@]} - 4 )) extra allocation node(s) idle."
fi

export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_FILE_LOCKING=1
export NUM_GPUS=1 MAX_BATCH_SIZE=128 PYTHONUNBUFFERED=1

SEARCH_URL=http://$SEARCH_NODE_IP:$SEARCH_PORT
SEARCH_LOG=$PROJECT_ROOT/logs/bc-search-${SLURM_JOB_ID:-idev}-${SEARCH_NODE}.log
SEARCH_ERROR_LOG=$PROJECT_ROOT/logs/bc-search-${SLURM_JOB_ID:-idev}-${SEARCH_NODE}.err
SEARCH_PID=
cleanup() {
  if [ -n "$SEARCH_PID" ]; then
    kill "$SEARCH_PID" 2>/dev/null || true
    wait "$SEARCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "4-node topology: retriever=$SEARCH_NODE; TRL trainer ranks=${TRAIN_NODES[*]}"
echo "Starting BC-P retriever on $SEARCH_NODE"
srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus-per-node=1 -w "$SEARCH_NODE" --input=none bash -lc "echo '[retriever-launch] host='\$(hostname)' time='\$(date -Is); source $CONDA_BASE/etc/profile.d/conda.sh; conda activate cxtgraph; cd $PROJECT_ROOT; export PYTHONPATH=$PROJECT_ROOT:\${PYTHONPATH:-}; exec python -u envs/search_server.py --model $EMBED_MODEL --host 0.0.0.0 --port $SEARCH_PORT --corpus Tevatron/browsecomp-plus-corpus --corpus-embedding-dataset miaolu3/browsecomp-plus" >"$SEARCH_LOG" 2>"$SEARCH_ERROR_LOG" &
SEARCH_PID=$!
echo "Retriever Slurm step PID: $SEARCH_PID"

SEARCH_LAUNCH_DEADLINE=$((SECONDS + BC_SEARCH_LAUNCH_TIMEOUT_SECONDS))
while [ "$SECONDS" -lt "$SEARCH_LAUNCH_DEADLINE" ] && [ ! -s "$SEARCH_LOG" ] && [ ! -s "$SEARCH_ERROR_LOG" ]; do
  if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ ! -s "$SEARCH_LOG" ] && [ ! -s "$SEARCH_ERROR_LOG" ]; then
  echo "ERROR: retriever Slurm task did not reach its launch sentinel within ${BC_SEARCH_LAUNCH_TIMEOUT_SECONDS}s." >&2
  ps -o pid,ppid,stat,wchan:32,etime,args -p "$SEARCH_PID" >&2 || true
  squeue --steps -j "${SLURM_JOB_ID:-}" -o '%.22i %.10T %.12M %.40R %.50j' >&2 || true
  exit 1
fi
tail -5 "$SEARCH_LOG" || true
tail -5 "$SEARCH_ERROR_LOG" || true

SEARCH_READY=0
SEARCH_DEADLINE=$((SECONDS + BC_SEARCH_TIMEOUT_SECONDS))
SEARCH_ATTEMPT=0
while [ "$SECONDS" -lt "$SEARCH_DEADLINE" ]; do
  SEARCH_ATTEMPT=$((SEARCH_ATTEMPT + 1))
  if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "$SEARCH_URL/health" >/dev/null 2>&1 && curl --noproxy '*' --connect-timeout 2 --max-time 10 -fsS -X POST -H 'Content-Type: application/json' -d '{"query":"Eiffel Tower","k":1}' "$SEARCH_URL/search" >/dev/null 2>&1; then
    SEARCH_READY=1
    echo "Retriever ready: $SEARCH_NODE"
    break
  fi
  if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
    echo "ERROR: retriever exited during startup." >&2
    tail -80 "$SEARCH_LOG" || true
    tail -80 "$SEARCH_ERROR_LOG" || true
    exit 1
  fi
  if [ $((SEARCH_ATTEMPT % 2)) -eq 0 ]; then
    echo "Waiting for retriever ($((BC_SEARCH_TIMEOUT_SECONDS - (SEARCH_DEADLINE - SECONDS)))s elapsed)"
    tail -5 "$SEARCH_LOG" || true
    tail -5 "$SEARCH_ERROR_LOG" || true
  fi
  sleep 1
done
if [ "$SEARCH_READY" != 1 ]; then
  echo "ERROR: BC-P retriever was not ready within ${BC_SEARCH_TIMEOUT_SECONDS}s." >&2
  tail -80 "$SEARCH_LOG" || true
  tail -80 "$SEARCH_ERROR_LOG" || true
  exit 1
fi

export LOCAL_SEARCH_URL=$SEARCH_URL
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
export RUN_TAG=${RUN_TAG:-trl-contextgraph-bc-qwen3-8b-lora32-${TOTAL_TRAINING_STEPS}step-3rank}
export RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
export RUN_NAME=${RUN_NAME:-${RUN_TAG}-${SLURM_JOB_ID:-local}-$RUN_TIMESTAMP}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SCRATCH:?SCRATCH must be set}/context-graph-trl-ckpts/$RUN_NAME}
export WANDB_TAGS=${WANDB_TAGS:-trl,contextgraph,browsecomp-plus,qwen3-8b,lora32,g16,${TOTAL_TRAINING_STEPS}step,ddp,3rank}
export TRL_NUM_MACHINES=3 TRL_NUM_PROCESSES=3 TRL_MASTER_ADDR=$TRAIN_MASTER_ADDR

echo "Launching distributed TRL ContextGraph BC-P: retriever=$LOCAL_SEARCH_URL trainers=$TRAIN_NODELIST model=$MODEL_PATH LoRA=32 steps=$TOTAL_TRAINING_STEPS"
srun --overlap --nodes=3 --ntasks=3 --ntasks-per-node=1 --gpus-per-node=1 -w "$TRAIN_NODELIST" --input=none bash "$PROJECT_ROOT/scripts/train_gsm8k_trl_agent_lora32.sh"
