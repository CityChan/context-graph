#!/bin/bash
# Build two replay-verified MiroVerse ContextGraph controller SFT rows with a
# DeepSeek-V4 teacher inside an existing four-GH200 Vista idev allocation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-deepseek_v4}
MODEL_ID=${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
TEACHER_TP=${TEACHER_TP:-4}
TEACHER_PORT=${TEACHER_PORT:-18000}
MAX_SAMPLES=${MAX_SAMPLES:-2}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
SAFETENSORS_LOAD_STRATEGY=${SAFETENSORS_LOAD_STRATEGY:-prefetch}
SAFETENSORS_PREFETCH_NUM_THREADS=${SAFETENSORS_PREFETCH_NUM_THREADS:-1}
DEEPSEEK_CUDA_HOME=${DEEPSEEK_CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8}
DEEPSEEK_MATH_LIB_ROOT=${DEEPSEEK_MATH_LIB_ROOT:-/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8}
DEEPSEEK_CUDA_MATH_INCLUDE=${DEEPSEEK_CUDA_MATH_INCLUDE:-$DEEPSEEK_MATH_LIB_ROOT/targets/sbsa-linux/include}
RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID:-idev}_miroverse_controller_smoke}

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
HF_HOME=${DEEPSEEK_HF_HOME:-$SCRATCH/hf_cache}
HF_HUB_CACHE=${DEEPSEEK_HF_HUB_CACHE:-$HF_HOME/hub}
MIROVERSE_ROOT=${MIROVERSE_ROOT:-$SCRATCH/datasets/MiroVerse-v0.1}
MIROVERSE_JSONL=${MIROVERSE_JSONL:-$MIROVERSE_ROOT/jsonl_sft/MiroVerse-MuSiQue.jsonl}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/miroverse_controller_smoke/$RUN_TAG}
OUTPUT_PARQUET=$ARTIFACT_ROOT/contextgraph_controller_smoke.parquet
OUTPUT_MANIFEST=$ARTIFACT_ROOT/manifest.json

resolve_snapshot() {
  local cache_name="models--${MODEL_ID//\//--}"
  local snapshot
  snapshot=$(find "$HF_HUB_CACHE/$cache_name/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
  if [ -n "$snapshot" ] && [ -s "$snapshot/config.json" ]; then
    printf '%s\n' "$snapshot"
    return 0
  fi
  return 1
}

if [ -n "${MODEL_PATH:-}" ]; then
  test -s "$MODEL_PATH/config.json" || { echo "ERROR: invalid MODEL_PATH=$MODEL_PATH"; exit 2; }
else
  MODEL_PATH=$(resolve_snapshot) || true
fi
test -n "${MODEL_PATH:-}" || { echo "ERROR: $MODEL_ID is not cached under $HF_HUB_CACHE"; exit 2; }
test -s "$MIROVERSE_JSONL" || {
  echo "ERROR: MiroVerse MuSiQue JSONL is missing: $MIROVERSE_JSONL"
  echo "After accepting the gated dataset license, run this on the login node:"
  echo "hf download miromind-ai/MiroVerse-v0.1 jsonl_sft/MiroVerse-MuSiQue.jsonl --repo-type dataset --local-dir \"$MIROVERSE_ROOT\""
  exit 2
}
test ! -e "$OUTPUT_PARQUET" || { echo "ERROR: smoke output already exists: $OUTPUT_PARQUET"; exit 2; }

mkdir -p "$PROJECT_ROOT/logs" "$ARTIFACT_ROOT"
cd "$PROJECT_ROOT"
export HF_HOME HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"
set -u

test -x "$DEEPSEEK_CUDA_HOME/bin/nvcc" || { echo "ERROR: CUDA compiler missing: $DEEPSEEK_CUDA_HOME/bin/nvcc"; exit 2; }
test -s "$DEEPSEEK_CUDA_MATH_INCLUDE/curand.h" || { echo "ERROR: CUDA math header missing: $DEEPSEEK_CUDA_MATH_INCLUDE/curand.h"; exit 2; }
export CUDA_HOME="$DEEPSEEK_CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=${DEEPSEEK_CC:-gcc}
export CXX=${DEEPSEEK_CXX:-g++}
export CUDAHOSTCXX=${DEEPSEEK_CUDAHOSTCXX:-g++}
export PATH="$CUDA_HOME/bin:$PATH"
DEEPSEEK_CUDA_MATH_LIB=${DEEPSEEK_CUDA_MATH_LIB:-$DEEPSEEK_MATH_LIB_ROOT/targets/sbsa-linux/lib}
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$DEEPSEEK_CUDA_MATH_LIB:$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$DEEPSEEK_CUDA_MATH_LIB:$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LIBRARY_PATH:-}"
export CPATH="$DEEPSEEK_CUDA_MATH_INCLUDE:$CUDA_HOME/include:${CPATH:-}"
export C_INCLUDE_PATH="$DEEPSEEK_CUDA_MATH_INCLUDE:$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$DEEPSEEK_CUDA_MATH_INCLUDE:$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"
export NVCC_PREPEND_FLAGS="-I$DEEPSEEK_CUDA_MATH_INCLUDE ${NVCC_PREPEND_FLAGS:-}"
export DG_JIT_CACHE_DIR=${DG_JIT_CACHE_DIR:-/tmp/contextgraph-deepgemm-$RUN_TAG}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/contextgraph-vllm-$RUN_TAG}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/tmp}
TORCH_GLOBAL_DEPS=$(python -c "import importlib.util,pathlib; s=importlib.util.find_spec('torch'); print(pathlib.Path(s.origin).parent/'lib'/'libtorch_global_deps.so')")
SERVER_LD_PRELOAD=${SERVER_LD_PRELOAD:-$TORCH_GLOBAL_DEPS}
test -s "$SERVER_LD_PRELOAD" || { echo "ERROR: server preload library missing: $SERVER_LD_PRELOAD"; exit 2; }

python -c "import openai,pandas,pyarrow,transformers,vllm; from packaging.version import Version; assert Version(vllm.__version__)>=Version('0.25.0'); print('smoke packages:',transformers.__version__,vllm.__version__)"
python -c "from agents.graph_controller import graph_action_schema; s=graph_action_schema([0,1],action_policy='structural'); assert s['properties']['action']['enum']==['merge','prune','add_edge']; print('controller schema: ok')"

mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NUM_NODES=${#NODELIST[@]}
test "$NUM_NODES" -eq "$EXPECTED_NUM_NODES" || { echo "ERROR: expected $EXPECTED_NUM_NODES nodes, got $NUM_NODES"; exit 2; }
test "$TEACHER_TP" -eq "$NUM_NODES" || { echo "ERROR: TEACHER_TP=$TEACHER_TP must equal NUM_NODES=$NUM_NODES"; exit 2; }
TEACHER_HEAD_NODE=${NODELIST[0]}
TEACHER_HEAD_IP=$(getent hosts "$TEACHER_HEAD_NODE" | awk '{print $1}')
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$TEACHER_HEAD_NODE,$TEACHER_HEAD_IP"
export no_proxy="$NO_PROXY"

echo "MiroVerse ContextGraph controller SFT smoke"
echo "Environment: $CONDA_ENV_NAME"
echo "Input: $MIROVERSE_JSONL"
echo "Teacher: $MODEL_ID at $MODEL_PATH"
echo "Nodes: ${NODELIST[*]} TP=$TEACHER_TP"
echo "Output: $OUTPUT_PARQUET"

STEP_PIDS=()
cleanup() {
  set +e
  for pid in "${STEP_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  for node in "${NODELIST[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $CONDA_ENV_NAME; ray stop --force >/dev/null 2>&1 || true" >/dev/null 2>&1 &
  done
  wait || true
}
trap cleanup EXIT

RAY_HEAD_LOG="$PROJECT_ROOT/logs/miroverse-controller-ray-head.$RUN_TAG.log"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $CONDA_ENV_NAME; ray stop --force >/dev/null 2>&1 || true; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export LD_PRELOAD=$SERVER_LD_PRELOAD OMP_NUM_THREADS=1; exec ray start --head --node-ip-address=$TEACHER_HEAD_IP --port=6379 --num-cpus=70 --num-gpus=1 --block" >"$RAY_HEAD_LOG" 2>&1 &
STEP_PIDS+=("$!")
sleep 8

for i in $(seq 1 $((NUM_NODES - 1))); do
  node=${NODELIST[$i]}
  worker_ip=$(getent hosts "$node" | awk '{print $1}')
  worker_log="$PROJECT_ROOT/logs/miroverse-controller-ray-worker-${i}.$RUN_TAG.log"
  srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $CONDA_ENV_NAME; ray stop --force >/dev/null 2>&1 || true; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export LD_PRELOAD=$SERVER_LD_PRELOAD OMP_NUM_THREADS=1; exec ray start --address=$TEACHER_HEAD_IP:6379 --node-ip-address=$worker_ip --num-cpus=70 --num-gpus=1 --block" >"$worker_log" 2>&1 &
  STEP_PIDS+=("$!")
done

RAY_READY=0
for _ in $(seq 1 180); do
  resources=$(srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $CONDA_ENV_NAME; RAY_ADDRESS=$TEACHER_HEAD_IP:6379 ray status 2>/dev/null" || true)
  if printf '%s\n' "$resources" | grep -Eq "0\.0/$TEACHER_TP\.0 GPU|$TEACHER_TP\.0 GPU"; then RAY_READY=1; break; fi
  sleep 2
done
test "$RAY_READY" -eq 1 || { echo "ERROR: Ray did not register all $TEACHER_TP GPUs"; exit 3; }

VLLM_LOG="$PROJECT_ROOT/logs/miroverse-controller-vllm.$RUN_TAG.log"
srun --overlap --nodes=1 --ntasks=1 -w "$TEACHER_HEAD_NODE" bash -lc "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate $CONDA_ENV_NAME; export CUDA_HOME=$CUDA_HOME CUDACXX=$CUDACXX CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX DG_JIT_CACHE_DIR=$DG_JIT_CACHE_DIR VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT FLASHINFER_WORKSPACE_BASE=$FLASHINFER_WORKSPACE_BASE; export PATH=$CUDA_HOME/bin:\$PATH LD_LIBRARY_PATH=$LD_LIBRARY_PATH LIBRARY_PATH=$LIBRARY_PATH CPATH=$CPATH C_INCLUDE_PATH=$C_INCLUDE_PATH CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH; export NVCC_PREPEND_FLAGS=\"$NVCC_PREPEND_FLAGS\"; export LD_PRELOAD=$SERVER_LD_PRELOAD OMP_NUM_THREADS=1 HF_HOME=$HF_HOME HF_HUB_CACHE=$HF_HUB_CACHE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_ADDRESS=$TEACHER_HEAD_IP:6379; exec vllm serve $MODEL_PATH --served-model-name $MODEL_ID --host 0.0.0.0 --port $TEACHER_PORT --distributed-executor-backend ray --tensor-parallel-size $TEACHER_TP --enable-expert-parallel --moe-backend auto --trust-remote-code --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8 --block-size 256 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --gpu-memory-utilization $GPU_MEMORY_UTILIZATION --safetensors-load-strategy $SAFETENSORS_LOAD_STRATEGY --safetensors-prefetch-num-threads $SAFETENSORS_PREFETCH_NUM_THREADS --enable-chunked-prefill" >"$VLLM_LOG" 2>&1 &
VLLM_STEP_PID=$!
STEP_PIDS+=("$VLLM_STEP_PID")

SERVER_READY=0
for attempt in $(seq 1 1800); do
  if curl --noproxy '*' -fsS "http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1/models" >/dev/null 2>&1; then SERVER_READY=1; break; fi
  if ! kill -0 "$VLLM_STEP_PID" 2>/dev/null; then
    wait "$VLLM_STEP_PID" || true
    echo "ERROR: vLLM exited during startup; tail of $VLLM_LOG follows"
    tail -n 120 "$VLLM_LOG" || true
    exit 3
  fi
  if [ $((attempt % 30)) -eq 0 ]; then echo "Waiting for vLLM: ${attempt}s elapsed"; tail -n 1 "$VLLM_LOG" || true; fi
  sleep 1
done
test "$SERVER_READY" -eq 1 || { echo "ERROR: vLLM health timeout; inspect $VLLM_LOG"; exit 3; }

python -u scripts/prepare_miroverse_contextgraph_controller_sft.py --input "$MIROVERSE_JSONL" --output "$OUTPUT_PARQUET" --manifest "$OUTPUT_MANIFEST" --max-samples "$MAX_SAMPLES" --source-subset MiroVerse-MuSiQue --teacher-base-url "http://$TEACHER_HEAD_IP:$TEACHER_PORT/v1" --teacher-model "$MODEL_ID" --max-output-tokens 256
python -c "import json,pandas as pd; p='$OUTPUT_PARQUET'; d=pd.read_parquet(p); m=json.load(open('$OUTPUT_MANIFEST')); assert $MAX_SAMPLES == 0 or len(d)==$MAX_SAMPLES; assert d['replay_valid'].all(); assert m['all_replay_valid']; assert all(x in {'merge','prune','add_edge'} for x in d['action']); print('MiroVerse controller SFT generation passed:',p,'rows=',len(d),'actions=',d['action'].value_counts().to_dict())"
