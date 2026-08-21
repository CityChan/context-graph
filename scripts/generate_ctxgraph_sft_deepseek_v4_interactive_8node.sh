#!/bin/bash
#SBATCH -J cg-sft-dsv4-mt
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/cg-sft-dsv4-mt.%A_%a.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/cg-sft-dsv4-mt.%A_%a.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A AST24021
#SBATCH --array=0-1

# Production multi-turn ContextGraph trajectory generation with DeepSeek-V4.
# Array task 0 curates ALFWorld train episodes; task 1 curates ScienceWorld
# train episodes. All eight GH200 nodes serve the MoE teacher; neither domain
# needs the BrowseComp retrieval server used by the nine-node search pipeline.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_ID=${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}
STUDENT_TOKENIZER_ID=${STUDENT_TOKENIZER_ID:-Qwen/Qwen3.6-27B}
SERVER_CONDA_ENV=${SERVER_CONDA_ENV:-deepseek_v4}
AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-8}
TEACHER_TP=${TEACHER_TP:-8}
TEACHER_PORT=${TEACHER_PORT:-18000}
MAX_SAMPLES=${MAX_SAMPLES:-100}
START_INDEX=${START_INDEX:-0}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_TURN=${MAX_TURN:-40}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-2048}
PROMPT_LENGTH=${PROMPT_LENGTH:-16384}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-24576}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-49152}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
REASONING_EFFORT=${REASONING_EFFORT:-non-thinking}
SCIENCEWORLD_VERSION=${SCIENCEWORLD_VERSION:-1.2.3}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
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
SHARED_HF_HOME=${SHARED_HF_HOME:-/work/09281/chc_1996/vista/cache}
SHARED_HF_HUB_CACHE=${SHARED_HF_HUB_CACHE:-$SHARED_HF_HOME/hub}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/deepseek_v4_flash_0731_interactive/$RUN_TAG/$DOMAIN}
RAW_OUTPUT_DIR=$ARTIFACT_ROOT/raw
SFT_OUTPUT=$ARTIFACT_ROOT/contextgraph_sft_train.parquet
SFT_VALIDATION_OUTPUT=$ARTIFACT_ROOT/contextgraph_sft_validation.parquet
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

if [ -n "${MODEL_PATH:-}" ]; then
  test -s "$MODEL_PATH/config.json" || { echo "ERROR: invalid MODEL_PATH=$MODEL_PATH"; exit 2; }
else
  MODEL_PATH=$(resolve_snapshot "$MODEL_ID" "$HF_HUB_CACHE" "$SHARED_HF_HUB_CACHE" "$SCRATCH/hf_cache") || true
fi
test -n "${MODEL_PATH:-}" || { echo "ERROR: $MODEL_ID is not cached"; exit 2; }

if [ -n "${STUDENT_TOKENIZER_PATH:-}" ]; then
  test -s "$STUDENT_TOKENIZER_PATH/config.json" || { echo "ERROR: invalid STUDENT_TOKENIZER_PATH=$STUDENT_TOKENIZER_PATH"; exit 2; }
else
  STUDENT_TOKENIZER_PATH=$(resolve_snapshot "$STUDENT_TOKENIZER_ID" "$HF_HUB_CACHE" "$SHARED_HF_HUB_CACHE" "$SCRATCH/hf_cache") || true
fi
test -n "${STUDENT_TOKENIZER_PATH:-}" || { echo "ERROR: $STUDENT_TOKENIZER_ID tokenizer is not cached"; exit 2; }

mkdir -p "$PROJECT_ROOT/logs" "$RAW_OUTPUT_DIR"
cd "$PROJECT_ROOT"
export HF_HOME HF_HUB_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$SERVER_CONDA_ENV"
set -u
python -c "import transformers, vllm; from packaging.version import Version; from transformers import AutoConfig; assert Version(vllm.__version__) >= Version('0.25.0'), 'DeepSeek-V4 requires vLLM >= 0.25.0'; c=AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True, local_files_only=True); print('server preflight:', 'transformers='+transformers.__version__, 'vllm='+vllm.__version__, 'model_type='+str(getattr(c, 'model_type', None)))"
VLLM_HELP=$(vllm serve --help=all 2>&1)
for required_flag in --distributed-executor-backend --tensor-parallel-size --enable-expert-parallel --kv-cache-dtype --tokenizer-mode --moe-backend; do
  printf '%s\n' "$VLLM_HELP" | grep -q -- "$required_flag" || { echo "ERROR: vLLM lacks $required_flag"; exit 2; }
done
if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "DeepSeek interactive SFT preflight passed: domain=$DOMAIN model=$MODEL_PATH tokenizer=$STUDENT_TOKENIZER_PATH"
  exit 0
fi

mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NUM_NODES=${#NODELIST[@]}
test "$NUM_NODES" -eq "$EXPECTED_NUM_NODES" || { echo "ERROR: expected $EXPECTED_NUM_NODES nodes, got $NUM_NODES"; exit 2; }
test "$TEACHER_TP" -eq "$NUM_NODES" || { echo "ERROR: TEACHER_TP=$TEACHER_TP must equal NUM_NODES=$NUM_NODES"; exit 2; }

TEACHER_HEAD_NODE=${NODELIST[0]}
TEACHER_HEAD_IP=$(getent hosts "$TEACHER_HEAD_NODE" | awk '{print $1}')
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$TEACHER_HEAD_NODE,$TEACHER_HEAD_IP"
export no_proxy=$NO_PROXY

echo "Domain: $DOMAIN"
echo "Model: $MODEL_ID at $MODEL_PATH"
echo "Student tokenizer: $STUDENT_TOKENIZER_PATH"
echo "Teacher nodes: ${NODELIST[*]}"
echo "Output: $ARTIFACT_ROOT"

STEP_PIDS=()
cleanup() {
  set +e
  for pid in "${STEP_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in "${NODELIST[@]}"; do srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; ray stop --force >/dev/null 2>&1 || true" >/dev/null 2>&1 & done
  wait || true
}
trap cleanup EXIT

RAY_HEAD_LOG="$PROJECT_ROOT/logs/cg-sft-dsv4-mt-ray-head.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}.log"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; ray stop --force >/dev/null 2>&1 || true; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE; exec ray start --head --node-ip-address=$TEACHER_HEAD_IP --port=6379 --num-cpus=70 --num-gpus=1 --block" >"$RAY_HEAD_LOG" 2>&1 &
STEP_PIDS+=("$!")
sleep 8

for i in $(seq 1 $((NUM_NODES - 1))); do
  node=${NODELIST[$i]}
  worker_ip=$(getent hosts "$node" | awk '{print $1}')
  worker_log="$PROJECT_ROOT/logs/cg-sft-dsv4-mt-ray-worker-${i}.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}.log"
  srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; ray stop --force >/dev/null 2>&1 || true; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE; exec ray start --address=$TEACHER_HEAD_IP:6379 --node-ip-address=$worker_ip --num-cpus=70 --num-gpus=1 --block" >"$worker_log" 2>&1 &
  STEP_PIDS+=("$!")
done

for _ in $(seq 1 180); do
  resources=$(srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; RAY_ADDRESS=$TEACHER_HEAD_IP:6379 ray status 2>/dev/null" || true)
  printf '%s\n' "$resources" | grep -Eq "0\.0/$TEACHER_TP\.0 GPU|$TEACHER_TP\.0 GPU" && break
  sleep 2
done

VLLM_LOG="$PROJECT_ROOT/logs/cg-sft-dsv4-mt-vllm.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}.log"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_ADDRESS=$TEACHER_HEAD_IP:6379; exec vllm serve $MODEL_PATH --served-model-name $MODEL_ID --host 0.0.0.0 --port $TEACHER_PORT --distributed-executor-backend ray --tensor-parallel-size $TEACHER_TP --enable-expert-parallel --moe-backend auto --trust-remote-code --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8 --block-size 256 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --enable-chunked-prefill" >"$VLLM_LOG" 2>&1 &
STEP_PIDS+=("$!")

for _ in $(seq 1 1800); do
  curl --noproxy '*' -fsS "http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done
curl --noproxy '*' -fsS "http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1/models" >/dev/null || { echo "ERROR: vLLM health timeout; inspect $VLLM_LOG"; exit 3; }

set +u
conda activate "$AGENT_CONDA_ENV"
set -u
if [ "$DOMAIN" = "alfworld" ]; then
  export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
  test -d "$ALFWORLD_DATA/json_2.1.1/train" || { echo "ERROR: ALFWorld train data missing under $ALFWORLD_DATA"; exit 2; }
  python scripts/make_alfworld_data.py --mode real --n_train "$((START_INDEX + MAX_SAMPLES))" --n_val 1 --seed 42
  DATA_PATH=data/alfworld_graph_real_train.parquet
  WORKFLOW=alfworld_graph
else
  if [ ! -s "$SCIENCEWORLD_DEPS/scienceworld/scienceworld.jar" ]; then
    mkdir -p "$SCIENCEWORLD_DEPS"
    python -m pip install --target "$SCIENCEWORLD_DEPS" --no-deps "scienceworld==$SCIENCEWORLD_VERSION" py4j
  fi
  export PYTHONPATH="$SCIENCEWORLD_DEPS:$PROJECT_ROOT:${PYTHONPATH:-}"
  java -version
  python scripts/make_scienceworld_data.py --n-train "$((START_INDEX + MAX_SAMPLES))" --n-val 1 --seed 42
  DATA_PATH=data/scienceworld_graph_train.parquet
  WORKFLOW=scienceworld_graph
fi

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL="http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1"
python scripts/eval_interactive.py --data-path "$DATA_PATH" --output-dir "$RAW_OUTPUT_DIR" --workflow "$WORKFLOW" --model-name "$MODEL_ID" --tokenizer-name "$STUDENT_TOKENIZER_PATH" --max-samples "$MAX_SAMPLES" --start-index "$START_INDEX" --num-workers "$NUM_WORKERS" --prompt-length "$PROMPT_LENGTH" --response-length "$RESPONSE_LENGTH" --max-turn "$MAX_TURN" --max-session 4 --branch-len 8192 --turn-max-new-tokens "$TURN_MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P" --reasoning-effort "$REASONING_EFFORT" --save-messages

RESULT_FILE=$(find "$RAW_OUTPUT_DIR" -maxdepth 1 -name 'interactive_results_*.json' -type f | sort | tail -n 1)
test -n "$RESULT_FILE" || { echo "ERROR: evaluator produced no result JSON"; exit 3; }

python scripts/build_contextgraph_sft.py "$RESULT_FILE" --output "$SFT_OUTPUT" --validation-output "$SFT_VALIDATION_OUTPUT" --validation-fraction 0.05 --min-task-reward 1.0 --min-valid-graph-ops 1 --min-structural-graph-ops 1 --max-invalid-graph-ops 0 --require-graph-trace --min-graph-quality-score 1.0 --max-redundant-graph-ops 0 --teacher-provider local_vllm --teacher-model "$MODEL_ID"
echo "DeepSeek interactive ContextGraph SFT generation complete: domain=$DOMAIN output=$SFT_OUTPUT"
