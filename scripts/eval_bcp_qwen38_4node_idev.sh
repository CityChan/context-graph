#!/bin/bash
# Existing allocation only. Run once per allocation: foldagent or contextgraph.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export SERVER_CONDA_ENV=${SERVER_CONDA_ENV:-deepseek_v4}
export AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
export CONDA_SH=${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}
export MODEL_ID=Qwen/Qwen3.8-27B
export SEARCH_PORT=${SEARCH_PORT:-18999} MODEL_PORT=${MODEL_PORT:-18000}
export SAMPLES=${SAMPLES:-8} SEED=${SEED:-42} WORKERS=${WORKERS:-2}
export MEMORY_MODE=${MEMORY_MODE:-repaired}
export DATA_PATH=${DATA_PATH:-$PROJECT_ROOT/data/bc_test.parquet}
export SEARCH_HF_HOME=${SEARCH_HF_HOME:-/work/09281/chc_1996/vista/cache}
export SEARCH_HF_HUB_CACHE=${SEARCH_HF_HUB_CACHE:-$SEARCH_HF_HOME/hub}
activate() {
  set +u
  source "$CONDA_SH"
  conda activate "$1"
  set -u
}
cd "$PROJECT_ROOT"

# Internal roles use inherited, validated paths; never submit another Slurm job.
case "${1:-}" in
  _server)
    activate "$SERVER_CONDA_ENV"
    # Vista's module environment can export CC=nvc. FlashInfer uses CC as
    # nvcc's -ccbin, so CUDAHOSTCXX alone does not fix its JIT compiler choice.
    export CC="${SERVER_CC:-gcc}" CXX="${SERVER_CXX:-g++}"
    CC=$(command -v "$CC")
    CXX=$(command -v "$CXX")
    export CC CXX CUDAHOSTCXX="$CXX" NVCC_CCBIN="$CXX"
    echo "Server compilers: CC=$CC CXX=$CXX CUDAHOSTCXX=$CUDAHOSTCXX"
    # Same Vista TLS workaround as the DeepSeek server launcher. Apply before
    # importing torch/vLLM; architecture-inspection subprocesses inherit it.
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    TORCH_GLOBAL_DEPS=$(python -c 'import importlib.util, pathlib; s=importlib.util.find_spec("torch"); print(pathlib.Path(s.origin).parent / "lib" / "libtorch_global_deps.so")')
    [ -s "$TORCH_GLOBAL_DEPS" ] || { echo "Missing PyTorch preload library: $TORCH_GLOBAL_DEPS"; exit 2; }
    export LD_PRELOAD="$TORCH_GLOBAL_DEPS${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "Server TLS setup: torch global deps preloaded; CPU thread limits=1"
    export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    python -c 'import os, transformers, vllm; from transformers import AutoConfig; c=AutoConfig.from_pretrained(os.environ["MODEL_PATH"], local_files_only=True); print("Server versions:", transformers.__version__, vllm.__version__, "architecture:", c.architectures, flush=True)'
    # Exercise the failing CUDA sampling path before loading the 27B weights.
    python -c 'import torch, flashinfer; x=torch.randn(2, 32, device="cuda", dtype=torch.float32); y=flashinfer.sampling.top_k_top_p_sampling_from_logits(x, 4, 0.9); torch.cuda.synchronize(); assert y.shape == (2,); print("FlashInfer sampling preflight passed", flush=True)'
    exec vllm serve "$MODEL_PATH" --served-model-name "$MODEL_ID" --host 0.0.0.0 --port "$MODEL_PORT" --tensor-parallel-size 1 --language-model-only --dtype bfloat16 --max-model-len 32768 --max-num-seqs "$WORKERS" --gpu-memory-utilization 0.90 --enable-chunked-prefill --generation-config vllm --seed "$SEED"
    ;;
  _search)
    activate "$AGENT_CONDA_ENV"
    export HF_HOME="$SEARCH_HF_HOME" HF_HUB_CACHE="$SEARCH_HF_HUB_CACHE"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NUM_GPUS=1 MAX_BATCH_SIZE=64
    exec python -u envs/search_server.py --model Qwen/Qwen3-Embedding-8B --host 0.0.0.0 --port "$SEARCH_PORT" --corpus Tevatron/browsecomp-plus-corpus --corpus-embedding-dataset miaolu3/browsecomp-plus
    ;;
  _eval)
    activate "$AGENT_CONDA_ENV"
    export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STRUCTURED_MEMORY_ENABLED=0
    unset QWEN_ENABLE_THINKING
    exec python -u scripts/eval_bcp_qwen38.py --method "$METHOD" --memory-mode "$MEMORY_MODE" --model-path "$MODEL_PATH" --endpoint "$2" --rank "$3" --data "$DATA_PATH" --samples "$SAMPLES" --seed "$SEED" --workers "$WORKERS" --output "$RUN_ROOT"
    ;;
esac

export METHOD=${1:?Usage: bash scripts/eval_bcp_qwen38_4node_idev.sh foldagent|contextgraph}
case "$METHOD" in foldagent|contextgraph) ;; *) echo "Invalid method: $METHOD"; exit 2 ;; esac
: "${SLURM_JOB_ID:?Run inside an existing four-node idev}"
: "${SLURM_JOB_NODELIST:?Missing allocation nodes}"
: "${SCRATCH:?Missing Vista scratch directory}"
if [ -n "${EXPECTED_JOB_ID:-}" ] && [ "$SLURM_JOB_ID" != "$EXPECTED_JOB_ID" ]; then
  echo "Wrong allocation: $SLURM_JOB_ID (expected $EXPECTED_JOB_ID)"; exit 2
fi
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#NODES[@]}" -eq 4 ] || { echo "Exactly four nodes required"; exit 2; }
mkdir -p logs
exec 9>"logs/bcp-qwen38-${SLURM_JOB_ID}.lock"
flock -n 9 || { echo "This allocation already has a Qwen3.8 evaluation"; exit 2; }
export RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/outputs/bcp-qwen38-${METHOD}-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT" # Do not overwrite or merge an earlier run.
exec > >(tee -a "$RUN_ROOT/suite.log") 2>&1
echo "Method=$METHOD memory=$MEMORY_MODE job=$SLURM_JOB_ID samples=$SAMPLES seed=$SEED"
echo "Output=$RUN_ROOT; allocation time limit still applies."
git rev-parse HEAD | tee "$RUN_ROOT/commit.txt"
git diff --exit-code HEAD -- agents envs scripts verl >/dev/null || { echo "Tracked code changes: commit/sync before paired evaluation"; exit 2; }
squeue -j "$SLURM_JOB_ID" -o '%.18i %.10L %.6D %N'

activate "$AGENT_CONDA_ENV"
if [ -f "${WORK:-/work/09281/chc_1996/vista}/.openai_env" ]; then
  source "${WORK:-/work/09281/chc_1996/vista}/.openai_env"
fi
[ -n "${OPENAI_API_KEY:-}" ] && [ "$OPENAI_API_KEY" != dummy ] || { echo "Missing real judge credentials"; exit 2; }
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano}
# OPENAI_BASE_URL belongs to the judge, never to the local model.
if [ -n "${JUDGE_BASE_URL:-}" ]; then
  export OPENAI_BASE_URL="$JUDGE_BASE_URL"
else
  unset OPENAI_BASE_URL
fi
export OPENAI_API_KEY
unset OPENAI_URL
export MODEL_PATH=${MODEL_PATH:-}
MODEL_PATH=$(python -c 'import os,json; from pathlib import Path; from huggingface_hub import snapshot_download; scratch=Path(os.environ["SCRATCH"]).resolve(); p=Path(os.environ["MODEL_PATH"] or snapshot_download(os.environ["MODEL_ID"], cache_dir=str(scratch/"hf_cache"/"hub"), local_files_only=True)).resolve(); assert p.is_relative_to(scratch), "Model must be under SCRATCH"; idx=json.loads((p/"model.safetensors.index.json").read_text()); shards=set(idx["weight_map"].values()); assert shards and all((p/s).is_file() and (p/s).stat().st_size>0 for s in shards), "Incomplete checkpoint"; print(p)')
export MODEL_PATH
echo "Resolved HF checkpoint: $MODEL_PATH"
python -c 'import os,pandas as pd; from scripts.eval_bcp_qwen38 import select_indices; from scripts.eval_gaia import _process_item_for_workflow; from transformers import AutoTokenizer; f=pd.read_parquet(os.environ["DATA_PATH"]); ids=select_indices(len(f),int(os.environ["SAMPLES"]),int(os.environ["SEED"])); assert len(ids)>=3; t=AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"],local_files_only=True); [_process_item_for_workflow(w) for w in ("search_branch","search_graph")]; print("Preflight rows:",len(f),"selected:",ids,"tokenizer:",type(t).__name__)'
python -c 'import os; from types import SimpleNamespace; from transformers import AutoTokenizer; from scripts.eval_bcp_qwen38 import config_for,tokenizer_preflight; os.environ.pop("QWEN_ENABLE_THINKING",None); t=AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"],local_files_only=True); c=config_for(SimpleNamespace(method=os.environ["METHOD"],memory_mode=os.environ["MEMORY_MODE"])); tokenizer_preflight(t,c.actor_rollout_ref.rollout); print("Observation tokenizer preflight passed")'
python -c 'import os; from openai import OpenAI; c=OpenAI(timeout=60,max_retries=1); r=c.chat.completions.create(model=os.environ["JUDGE_MODEL"],messages=[{"role":"user","content":"Reply OK."}]); assert r.choices; c.close(); print("Judge API preflight passed")'
sha256sum "$DATA_PATH" > "$RUN_ROOT/data.sha256"

SELF="$PROJECT_ROOT/scripts/eval_bcp_qwen38_4node_idev.sh"
SERVICE_PIDS=()
EVAL_PIDS=()
cleanup() {
  set +e
  for pid in "${EVAL_PIDS[@]}" "${SERVICE_PIDS[@]}"; do kill "$pid" 2>/dev/null; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
SEARCH_IP=$(getent ahostsv4 "${NODES[0]}" | awk 'NR==1 {print $1}')
[ -n "$SEARCH_IP" ] || { echo "Cannot resolve search node"; exit 2; }
export LOCAL_SEARCH_URL="http://$SEARCH_IP:$SEARCH_PORT"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,${NODES[0]},$SEARCH_IP"
MODEL_IPS=()
for node in "${NODES[@]:1}"; do
  ip=$(getent ahostsv4 "$node" | awk 'NR==1 {print $1}')
  [ -n "$ip" ] || { echo "Cannot resolve $node"; exit 2; }
  MODEL_IPS+=("$ip")
  NO_PROXY="$NO_PROXY,$node,$ip"
done
export NO_PROXY no_proxy="$NO_PROXY"
srun --overlap -N 1 -n 1 -w "${NODES[0]}" bash "$SELF" _search >"$RUN_ROOT/search.log" 2>&1 &
SERVICE_PIDS+=("$!")
for rank in 0 1 2; do
  srun --overlap -N 1 -n 1 -w "${NODES[$((rank+1))]}" bash "$SELF" _server >"$RUN_ROOT/model-$rank.log" 2>&1 &
  SERVICE_PIDS+=("$!")
done
wait_health() {
  local url=$1 pid=$2 label=$3
  for ((attempt=0; attempt<1200; attempt++)); do
    kill -0 "$pid" 2>/dev/null || { echo "$label exited; last 120 log lines:"; tail -n 120 "$RUN_ROOT/$label.log" || true; return 1; }
    if curl --noproxy '*' --max-time 3 -fsS "$url" >/dev/null 2>&1; then return 0; fi
    if (( attempt % 60 == 0 )); then echo "Waiting for $label ($attempt seconds)..."; fi
    sleep 1
  done
  echo "$label startup timed out"; return 1
}
wait_health "$LOCAL_SEARCH_URL/health" "${SERVICE_PIDS[0]}" search
for rank in 0 1 2; do
  wait_health "http://${MODEL_IPS[$rank]}:$MODEL_PORT/health" "${SERVICE_PIDS[$((rank+1))]}" "model-$rank"
done
for rank in 0 1 2; do
  srun --overlap -N 1 -n 1 -w "${NODES[$((rank+1))]}" bash "$SELF" _eval "http://${MODEL_IPS[$rank]}:$MODEL_PORT" "$rank" >"$RUN_ROOT/eval-$rank.log" 2>&1 &
  EVAL_PIDS+=("$!")
done
echo "Three evaluation shards running; per-task results stream into $RUN_ROOT/results-*.jsonl"
rc=0
for pid in "${EVAL_PIDS[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then echo "Evaluation shard failed; inspect eval-*.log"; exit 1; fi
python scripts/eval_bcp_qwen38.py --merge --output "$RUN_ROOT"
echo "BCP_EVAL_COMPLETE method=$METHOD output=$RUN_ROOT"
