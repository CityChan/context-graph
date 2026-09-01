#!/bin/bash
#SBATCH -J gen-cg-sft-dsv4
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/gen-cg-sft-dsv4.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/gen-cg-sft-dsv4.%j.err
#SBATCH -p gh
#SBATCH -N 9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 08:00:00
#SBATCH -A AST24021

# Generate executor-verified ContextGraph SFT trajectories with the open-source
# DeepSeek-V4-Flash-0731 checkpoint. Eight GH200 nodes serve one TP=8 vLLM
# engine and the ninth node serves BrowseComp-Plus retrieval.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_ID=${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}
SERVER_CONDA_ENV=${SERVER_CONDA_ENV:-deepseek_v4}
AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
DEEPSEEK_CUDA_HOME=${DEEPSEEK_CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8}
DEEPSEEK_MATH_LIB_ROOT=${DEEPSEEK_MATH_LIB_ROOT:-/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8}
DEEPSEEK_CUDA_MATH_INCLUDE=${DEEPSEEK_CUDA_MATH_INCLUDE:-$DEEPSEEK_MATH_LIB_ROOT/targets/sbsa-linux/include}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-9}
TEACHER_TP=${TEACHER_TP:-8}
TEACHER_PORT=${TEACHER_PORT:-18000}
SEARCH_PORT=${SEARCH_PORT:-18999}
MAX_SAMPLES=${MAX_SAMPLES:-25}
START_INDEX=${START_INDEX:-0}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_TURN=${MAX_TURN:-24}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-2048}
PROMPT_LENGTH=${PROMPT_LENGTH:-16384}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-16384}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
SAFETENSORS_LOAD_STRATEGY=${SAFETENSORS_LOAD_STRATEGY:-prefetch}
SAFETENSORS_PREFETCH_NUM_THREADS=${SAFETENSORS_PREFETCH_NUM_THREADS:-1}
REASONING_EFFORT=${REASONING_EFFORT:-non-thinking}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
DATA_PATH=${DATA_PATH:-data/bc_train.parquet}
ALLOW_EVAL_DATA=${ALLOW_EVAL_DATA:-0}
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}
STUDENT_TOKENIZER_ID=${STUDENT_TOKENIZER_ID:-Qwen/Qwen3-8B}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-local}}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
PREFLIGHT_TIMEOUT_SECONDS=${PREFLIGHT_TIMEOUT_SECONDS:-180}
FULL_POLICY_CURATOR=${FULL_POLICY_CURATOR:-0}

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
HF_HOME=${DEEPSEEK_HF_HOME:-$SCRATCH/hf_cache}
HF_HUB_CACHE=${DEEPSEEK_HF_HUB_CACHE:-$HF_HOME/hub}
SCRATCH_HF_HUB_CACHE=${SCRATCH_HF_HUB_CACHE:-$SCRATCH/hf_cache/hub}
SEARCH_HF_HOME=${SEARCH_HF_HOME:-/work/09281/chc_1996/vista/cache}
SEARCH_HF_HUB_CACHE=${SEARCH_HF_HUB_CACHE:-$SEARCH_HF_HOME/hub}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/deepseek_v4_flash_0731/$RUN_TAG}
RAW_OUTPUT_DIR=$ARTIFACT_ROOT/raw
SFT_OUTPUT=$ARTIFACT_ROOT/contextgraph_sft_train.parquet
SFT_VALIDATION_OUTPUT=$ARTIFACT_ROOT/contextgraph_sft_validation.parquet
SFT_MANIFEST=$ARTIFACT_ROOT/manifest.json

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
  if [ ! -s "$MODEL_PATH/config.json" ]; then
    echo "ERROR: MODEL_PATH does not contain config.json: $MODEL_PATH"
    exit 2
  fi
else
  for candidate in "$SCRATCH/DeepSeek-V4-Flash-0731" "$SCRATCH/models/DeepSeek-V4-Flash-0731" "$SCRATCH/models/deepseek-ai/DeepSeek-V4-Flash-0731"; do
    if [ -s "$candidate/config.json" ]; then
      MODEL_PATH=$candidate
      break
    fi
  done
  if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH=$(resolve_snapshot "$MODEL_ID" "$HF_HUB_CACHE" "$SCRATCH_HF_HUB_CACHE" "$SCRATCH/hf_cache") || true
  fi
fi
if [ -z "${MODEL_PATH:-}" ]; then
  echo "ERROR: could not find $MODEL_ID under SCRATCH=$SCRATCH"
  echo "Expected a direct model directory or $HF_HUB_CACHE/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/<revision>"
  echo "Override with MODEL_PATH=/absolute/path/to/DeepSeek-V4-Flash-0731"
  exit 2
fi
require_scratch_path MODEL_PATH "$MODEL_PATH"

if [ -n "${STUDENT_TOKENIZER_PATH:-}" ]; then
  if [ ! -s "$STUDENT_TOKENIZER_PATH/config.json" ] && [ ! -s "$STUDENT_TOKENIZER_PATH/tokenizer_config.json" ]; then
    echo "ERROR: invalid STUDENT_TOKENIZER_PATH: $STUDENT_TOKENIZER_PATH"
    exit 2
  fi
else
  STUDENT_TOKENIZER_PATH=$(resolve_snapshot "$STUDENT_TOKENIZER_ID" "$HF_HUB_CACHE" "$SEARCH_HF_HUB_CACHE") || true
fi
if [ -z "${STUDENT_TOKENIZER_PATH:-}" ]; then
  echo "ERROR: could not find student tokenizer $STUDENT_TOKENIZER_ID"
  echo "Set STUDENT_TOKENIZER_PATH to a cached Qwen tokenizer with a Jinja chat template."
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
DEEPSEEK_CACHE_TAG=${SLURM_JOB_ID:-local}_0
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/tmp/contextgraph-deepgemm-$DEEPSEEK_CACHE_TAG}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/contextgraph-vllm-$DEEPSEEK_CACHE_TAG}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp}
echo "DeepSeek toolchain: nvcc=$CUDACXX host_cxx=$(command -v "$CXX")"
echo "CUDA math headers: $DEEPSEEK_CUDA_MATH_INCLUDE"
echo "Node-local JIT caches: DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE"
echo "Preflight: importing server packages and reading model config"
timeout "$PREFLIGHT_TIMEOUT_SECONDS" python -u -c "print('preflight stage: import transformers', flush=True); import transformers; print('preflight stage: import vllm', flush=True); import vllm; print('preflight stage: read model config', flush=True); from packaging.version import Version; from transformers import AutoConfig; assert Version(vllm.__version__) >= Version('0.25.0'), 'DeepSeek-V4-Flash-0731 requires vLLM >= 0.25.0'; c=AutoConfig.from_pretrained('$MODEL_PATH', trust_remote_code=True, local_files_only=True); print('server preflight:', 'transformers='+transformers.__version__, 'vllm='+vllm.__version__, 'model_type='+str(getattr(c, 'model_type', None)), flush=True)" || { echo "ERROR: server package/model preflight failed or exceeded ${PREFLIGHT_TIMEOUT_SECONDS}s"; exit 2; }
# vLLM 0.27 uses paged/grouped CLI help; plain --help intentionally omits
# model and parallelism options.
echo "Preflight: checking required vLLM CLI flags"
if ! VLLM_HELP=$(timeout "$PREFLIGHT_TIMEOUT_SECONDS" vllm serve --help=all 2>&1); then
  echo "ERROR: vLLM CLI preflight failed or exceeded ${PREFLIGHT_TIMEOUT_SECONDS}s"
  exit 2
fi
for required_flag in --distributed-executor-backend --tensor-parallel-size --enable-expert-parallel --kv-cache-dtype --tokenizer-mode --moe-backend --safetensors-load-strategy --safetensors-prefetch-num-threads; do
  if ! printf '%s\n' "$VLLM_HELP" | grep -q -- "$required_flag"; then
    echo "ERROR: $SERVER_CONDA_ENV vLLM does not support $required_flag"
    echo "Install a DeepSeek-V4-compatible vLLM build in the dedicated server environment."
    exit 2
  fi
done
if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "DeepSeek-V4 SFT preflight passed."
  echo "Model path: $MODEL_PATH"
  echo "Student tokenizer: $STUDENT_TOKENIZER_PATH"
  exit 0
fi

mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NUM_NODES=${#NODELIST[@]}
if [ "$NUM_NODES" -ne "$EXPECTED_NUM_NODES" ]; then
  echo "ERROR: expected $EXPECTED_NUM_NODES nodes, got $NUM_NODES"
  exit 2
fi
if [ "$TEACHER_TP" -ne $((NUM_NODES - 1)) ]; then
  echo "ERROR: TEACHER_TP=$TEACHER_TP must equal model node count $((NUM_NODES - 1))"
  exit 2
fi

SEARCH_NODE=${NODELIST[0]}
SEARCH_NODE_IP=$(getent hosts "$SEARCH_NODE" | awk '{print $1}')
TEACHER_HEAD_NODE=${NODELIST[1]}
TEACHER_HEAD_IP=$(getent hosts "$TEACHER_HEAD_NODE" | awk '{print $1}')
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$SEARCH_NODE,$SEARCH_NODE_IP,$TEACHER_HEAD_NODE,$TEACHER_HEAD_IP"
export no_proxy=$NO_PROXY

echo "Model ID:   $MODEL_ID"
echo "Model path: $MODEL_PATH"
echo "Tokenizer:  $STUDENT_TOKENIZER_PATH"
echo "Server env: $SERVER_CONDA_ENV; TP=$TEACHER_TP; model nodes=${NODELIST[*]:1}"
echo "Search:     $SEARCH_NODE ($SEARCH_NODE_IP:$SEARCH_PORT)"
echo "Safetensors load strategy: $SAFETENSORS_LOAD_STRATEGY (threads=$SAFETENSORS_PREFETCH_NUM_THREADS)"
echo "Output:     $ARTIFACT_ROOT"

STEP_PIDS=()
cleanup() {
  set +e
  for pid in "${STEP_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in "${NODELIST[@]:1}"; do srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; ray stop --force >/dev/null 2>&1 || true" >/dev/null 2>&1 & done
  wait || true
}
trap cleanup EXIT

SEARCH_LOG="$PROJECT_ROOT/logs/gen-cg-sft-dsv4-search.${SLURM_JOB_ID:-local}.log"
srun --overlap --nodes=1 --ntasks=1 -w "$SEARCH_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $AGENT_CONDA_ENV; cd $PROJECT_ROOT; export HF_HOME=$SEARCH_HF_HOME HF_HUB_CACHE=$SEARCH_HF_HUB_CACHE NUM_GPUS=1 MAX_BATCH_SIZE=128; exec python -u envs/search_server.py --model $EMBED_MODEL --host 0.0.0.0 --port $SEARCH_PORT --corpus Tevatron/browsecomp-plus-corpus --corpus-embedding-dataset miaolu3/browsecomp-plus" >"$SEARCH_LOG" 2>&1 &
STEP_PIDS+=("$!")

for _ in $(seq 1 600); do
  if curl --noproxy '*' -fsS "http://$SEARCH_NODE_IP:$SEARCH_PORT/health" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl --noproxy '*' -fsS "http://$SEARCH_NODE_IP:$SEARCH_PORT/health" >/dev/null

RAY_HEAD_LOG="$PROJECT_ROOT/logs/gen-cg-sft-dsv4-ray-head.${SLURM_JOB_ID:-local}.log"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; ray stop --force >/dev/null 2>&1 || true; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE; exec ray start --head --node-ip-address=$TEACHER_HEAD_IP --port=6379 --num-cpus=70 --num-gpus=1 --block" >"$RAY_HEAD_LOG" 2>&1 &
STEP_PIDS+=("$!")
sleep 8

for i in $(seq 2 $((NUM_NODES - 1))); do
  node=${NODELIST[$i]}
  worker_ip=$(getent hosts "$node" | awk '{print $1}')
  worker_log="$PROJECT_ROOT/logs/gen-cg-sft-dsv4-ray-worker-${i}.${SLURM_JOB_ID:-local}.log"
  srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; ray stop --force >/dev/null 2>&1 || true; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE; exec ray start --address=$TEACHER_HEAD_IP:6379 --node-ip-address=$worker_ip --num-cpus=70 --num-gpus=1 --block" >"$worker_log" 2>&1 &
  STEP_PIDS+=("$!")
done

for _ in $(seq 1 180); do
  resources=$(srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; RAY_ADDRESS=$TEACHER_HEAD_IP:6379 ray status 2>/dev/null" || true)
  if printf '%s\n' "$resources" | grep -Eq "0\.0/$TEACHER_TP\.0 GPU|$TEACHER_TP\.0 GPU"; then break; fi
  sleep 2
done

VLLM_LOG="$PROJECT_ROOT/logs/gen-cg-sft-dsv4-vllm.${SLURM_JOB_ID:-local}.log"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $SERVER_CONDA_ENV; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_ADDRESS=$TEACHER_HEAD_IP:6379; exec vllm serve $MODEL_PATH --served-model-name $MODEL_ID --host 0.0.0.0 --port $TEACHER_PORT --distributed-executor-backend ray --tensor-parallel-size $TEACHER_TP --enable-expert-parallel --moe-backend auto --trust-remote-code --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8 --block-size 256 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --safetensors-load-strategy $SAFETENSORS_LOAD_STRATEGY --safetensors-prefetch-num-threads $SAFETENSORS_PREFETCH_NUM_THREADS --enable-chunked-prefill" >"$VLLM_LOG" 2>&1 &
STEP_PIDS+=("$!")

for _ in $(seq 1 1800); do
  if curl --noproxy '*' -fsS "http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1/models" >/dev/null 2>&1; then break; fi
  sleep 1
done
curl --noproxy '*' -fsS "http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1/models" >/dev/null

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL="http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1"
export LOCAL_SEARCH_URL="http://$SEARCH_NODE_IP:$SEARCH_PORT"

srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $AGENT_CONDA_ENV; cd $PROJECT_ROOT; export OPENAI_API_KEY=dummy OPENAI_BASE_URL=$OPENAI_BASE_URL LOCAL_SEARCH_URL=$LOCAL_SEARCH_URL; python scripts/eval_gaia.py --data-path $DATA_PATH --output-dir $RAW_OUTPUT_DIR --workflow search_graph --model-name $MODEL_ID --tokenizer-name $STUDENT_TOKENIZER_PATH --num-workers $NUM_WORKERS --start-index $START_INDEX --max-samples $MAX_SAMPLES --prompt-length $PROMPT_LENGTH --response-length $RESPONSE_LENGTH --max-turn $MAX_TURN --max-session 8 --branch-len 8192 --turn-max-new-tokens $TURN_MAX_NEW_TOKENS --temperature $TEMPERATURE --top-p $TOP_P --reasoning-effort $REASONING_EFFORT --local-search-url $LOCAL_SEARCH_URL --must-search --save-messages"

RESULT_FILE=$(find "$RAW_OUTPUT_DIR" -maxdepth 1 -name 'gaia_results_*.json' -type f | sort | tail -n 1)
if [ -z "$RESULT_FILE" ]; then
  echo "ERROR: evaluator produced no result JSON"
  exit 3
fi

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$AGENT_CONDA_ENV"
set -u
if [ "$FULL_POLICY_CURATOR" = "1" ]; then
  python scripts/build_contextgraph_full_policy_sft.py "$RESULT_FILE" --output "$SFT_OUTPUT" --validation-output "$SFT_VALIDATION_OUTPUT" --manifest "$SFT_MANIFEST" --validation-fraction 0.05 --teacher-model "$MODEL_ID"
else
  python scripts/build_contextgraph_sft.py "$RESULT_FILE" --output "$SFT_OUTPUT" --validation-output "$SFT_VALIDATION_OUTPUT" --validation-fraction 0.05 --min-task-reward 1.0 --min-valid-graph-ops 1 --min-structural-graph-ops 1 --max-invalid-graph-ops 0 --require-graph-trace --min-graph-quality-score 1.0 --max-redundant-graph-ops 0 --teacher-provider local_vllm --teacher-model "$MODEL_ID"
fi
echo "DeepSeek ContextGraph SFT generation complete: $SFT_OUTPUT"
