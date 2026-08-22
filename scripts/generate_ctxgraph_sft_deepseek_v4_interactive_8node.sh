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
DEEPSEEK_CUDA_HOME=${DEEPSEEK_CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8}
DEEPSEEK_MATH_LIB_ROOT=${DEEPSEEK_MATH_LIB_ROOT:-/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8}
DEEPSEEK_CUDA_MATH_INCLUDE=${DEEPSEEK_CUDA_MATH_INCLUDE:-$DEEPSEEK_MATH_LIB_ROOT/targets/sbsa-linux/include}
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
PREFLIGHT_TIMEOUT_SECONDS=${PREFLIGHT_TIMEOUT_SECONDS:-180}
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
# Ignore a stale login-shell HF_HOME under /work. These dedicated overrides
# keep the large DeepSeek checkpoint on Vista scratch by construction.
HF_HOME=${DEEPSEEK_HF_HOME:-$SCRATCH/hf_cache}
HF_HUB_CACHE=${DEEPSEEK_HF_HUB_CACHE:-$HF_HOME/hub}
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

if [ -n "${STUDENT_TOKENIZER_PATH:-}" ]; then
  test -s "$STUDENT_TOKENIZER_PATH/config.json" || { echo "ERROR: invalid STUDENT_TOKENIZER_PATH=$STUDENT_TOKENIZER_PATH"; exit 2; }
else
  STUDENT_TOKENIZER_PATH=$(resolve_snapshot "$STUDENT_TOKENIZER_ID" "$HF_HUB_CACHE" "$SCRATCH/hf_cache") || true
fi
test -n "${STUDENT_TOKENIZER_PATH:-}" || { echo "ERROR: $STUDENT_TOKENIZER_ID tokenizer is not cached"; exit 2; }
require_scratch_path STUDENT_TOKENIZER_PATH "$STUDENT_TOKENIZER_PATH"

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
test -x "$DEEPSEEK_CUDA_HOME/bin/nvcc" || { echo "ERROR: CUDA compiler missing: $DEEPSEEK_CUDA_HOME/bin/nvcc"; exit 2; }
test -s "$DEEPSEEK_CUDA_MATH_INCLUDE/curand.h" || { echo "ERROR: CUDA math header missing: $DEEPSEEK_CUDA_MATH_INCLUDE/curand.h"; exit 2; }
export CUDA_HOME="$DEEPSEEK_CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=${DEEPSEEK_CC:-gcc}
export CXX=${DEEPSEEK_CXX:-g++}
export CUDAHOSTCXX=${DEEPSEEK_CUDAHOSTCXX:-g++}
command -v "$CC" >/dev/null || { echo "ERROR: C compiler not found: $CC"; exit 2; }
command -v "$CXX" >/dev/null || { echo "ERROR: C++ compiler not found: $CXX"; exit 2; }
DEEPSEEK_CUDA_MATH_LIB=${DEEPSEEK_CUDA_MATH_LIB:-$DEEPSEEK_MATH_LIB_ROOT/targets/sbsa-linux/lib}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$DEEPSEEK_CUDA_MATH_LIB:$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$DEEPSEEK_CUDA_MATH_LIB:$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export CPATH="$DEEPSEEK_CUDA_MATH_INCLUDE:$CUDA_HOME/include:${CPATH:-}"
export C_INCLUDE_PATH="$DEEPSEEK_CUDA_MATH_INCLUDE:$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$DEEPSEEK_CUDA_MATH_INCLUDE:$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export NVCC_PREPEND_FLAGS="-I$DEEPSEEK_CUDA_MATH_INCLUDE ${NVCC_PREPEND_FLAGS:-}"
# /tmp is node-local on Vista. A common path on a shared filesystem can make
# concurrent DeepGEMM JIT writers corrupt one another's cache entries.
DEEPSEEK_CACHE_TAG=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/tmp/contextgraph-deepgemm-$DEEPSEEK_CACHE_TAG}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/contextgraph-vllm-$DEEPSEEK_CACHE_TAG}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp}
echo "DeepSeek toolchain: nvcc=$CUDACXX host_cxx=$(command -v "$CXX")"
echo "CUDA math headers: $DEEPSEEK_CUDA_MATH_INCLUDE"
echo "Node-local JIT caches: DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"
TORCH_GLOBAL_DEPS=$(python -c "import importlib.util, pathlib; s=importlib.util.find_spec('torch'); print(pathlib.Path(s.origin).parent / 'lib' / 'libtorch_global_deps.so')")
SERVER_LD_PRELOAD=${SERVER_LD_PRELOAD:-$TORCH_GLOBAL_DEPS}
test -s "$SERVER_LD_PRELOAD" || { echo "ERROR: server preload library missing: $SERVER_LD_PRELOAD"; exit 2; }
echo "Server LD preload: $SERVER_LD_PRELOAD"
echo "Preflight: importing server packages and reading model config"
timeout "$PREFLIGHT_TIMEOUT_SECONDS" python -u -c "import transformers, vllm; from packaging.version import Version; from transformers import AutoConfig; assert Version(vllm.__version__) >= Version('0.25.0'), 'DeepSeek-V4 requires vLLM >= 0.25.0'; c=AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True, local_files_only=True); print('server preflight:', 'transformers='+transformers.__version__, 'vllm='+vllm.__version__, 'model_type='+str(getattr(c, 'model_type', None)))" || { echo "ERROR: server package/model preflight failed or exceeded ${PREFLIGHT_TIMEOUT_SECONDS}s"; exit 2; }
echo "Preflight: checking required vLLM CLI flags"
if ! VLLM_HELP=$(timeout "$PREFLIGHT_TIMEOUT_SECONDS" vllm serve --help=all 2>&1); then
  echo "ERROR: vLLM CLI preflight failed or exceeded ${PREFLIGHT_TIMEOUT_SECONDS}s"
  exit 2
fi
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
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; ray stop --force >/dev/null 2>&1 || true; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export LD_PRELOAD=$SERVER_LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE; exec ray start --head --node-ip-address=$TEACHER_HEAD_IP --port=6379 --num-cpus=70 --num-gpus=1 --block" >"$RAY_HEAD_LOG" 2>&1 &
STEP_PIDS+=("$!")
sleep 8

for i in $(seq 1 $((NUM_NODES - 1))); do
  node=${NODELIST[$i]}
  worker_ip=$(getent hosts "$node" | awk '{print $1}')
  worker_log="$PROJECT_ROOT/logs/cg-sft-dsv4-mt-ray-worker-${i}.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}.log"
  srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; ray stop --force >/dev/null 2>&1 || true; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export LD_PRELOAD=$SERVER_LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE; exec ray start --address=$TEACHER_HEAD_IP:6379 --node-ip-address=$worker_ip --num-cpus=70 --num-gpus=1 --block" >"$worker_log" 2>&1 &
  STEP_PIDS+=("$!")
done

for _ in $(seq 1 180); do
  resources=$(srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; RAY_ADDRESS=$TEACHER_HEAD_IP:6379 ray status 2>/dev/null" || true)
  printf '%s\n' "$resources" | grep -Eq "0\.0/$TEACHER_TP\.0 GPU|$TEACHER_TP\.0 GPU" && break
  sleep 2
done

VLLM_LOG="$PROJECT_ROOT/logs/cg-sft-dsv4-mt-vllm.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-0}.log"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export LD_PRELOAD=$SERVER_LD_PRELOAD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_ADDRESS=$TEACHER_HEAD_IP:6379; exec vllm serve $MODEL_PATH --served-model-name $MODEL_ID --host 0.0.0.0 --port $TEACHER_PORT --distributed-executor-backend ray --tensor-parallel-size $TEACHER_TP --enable-expert-parallel --moe-backend auto --trust-remote-code --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8 --block-size 256 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --enable-chunked-prefill" >"$VLLM_LOG" 2>&1 &
VLLM_STEP_PID=$!
STEP_PIDS+=("$!")

SERVER_READY=0
for attempt in $(seq 1 1800); do
  if curl --noproxy '*' -fsS "http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1/models" >/dev/null 2>&1; then
    SERVER_READY=1
    break
  fi
  if ! kill -0 "$VLLM_STEP_PID" 2>/dev/null; then
    wait "$VLLM_STEP_PID" || true
    echo "ERROR: vLLM exited during startup; tail of $VLLM_LOG follows"
    tail -n 120 "$VLLM_LOG" || true
    exit 3
  fi
  if [ $((attempt % 30)) -eq 0 ]; then
    echo "Waiting for vLLM health: ${attempt}s elapsed; log=$VLLM_LOG"
    tail -n 1 "$VLLM_LOG" 2>/dev/null | sed 's/^/vLLM latest: /' || true
  fi
  sleep 1
done
test "$SERVER_READY" -eq 1 || { echo "ERROR: vLLM health timeout; inspect $VLLM_LOG"; exit 3; }

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
