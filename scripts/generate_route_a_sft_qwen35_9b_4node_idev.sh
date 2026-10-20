#!/bin/bash
# Route A self-distillation: one policy server, optional BC-P retrieval server,
# CPU collection on a third node. Run only in an idle four-node allocation.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
cd "$PROJECT_ROOT"
: "${SLURM_JOB_ID:?Run inside an idle four-node idev allocation}"
: "${SCRATCH:?Missing SCRATCH}"
domain=${1:-scienceworld}
case "$domain" in bcp|scienceworld) ;; *) echo 'Expected bcp or scienceworld' >&2; exit 2 ;; esac
export CONDA_SH=${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}
export AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
set +u
source "$CONDA_SH"
conda activate "$AGENT_CONDA_ENV"
set -u
export MODEL_ID=Qwen/Qwen3.5-9B
export MODEL_PATH=${QWEN35_9B_MODEL_PATH:-}
MODEL_PATH=$(python -c 'import os; from pathlib import Path; from huggingface_hub import snapshot_download; p=Path(os.environ["MODEL_PATH"] or snapshot_download(os.environ["MODEL_ID"],cache_dir=os.environ["SCRATCH"]+"/hf_cache/hub",local_files_only=True)).resolve(); assert (p/"config.json").is_file(); print(p)')
export MODEL_PATH
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536} PROMPT_LENGTH=${PROMPT_LENGTH:-8192}
[[ "$CONTEXT_LENGTH" =~ ^[1-9][0-9]*$ && "$PROMPT_LENGTH" =~ ^[1-9][0-9]*$ ]] || exit 2
(( CONTEXT_LENGTH > PROMPT_LENGTH )) || exit 2
export MODEL_MAX_LEN=$CONTEXT_LENGTH AGENT_BUDGET_MODE=working_context
export SAMPLES=${SAMPLES:-3} CANDIDATES=${CANDIDATES:-2} SEED=${SEED:-42} WORKERS=${WORKERS:-1}
commit=$(git rev-parse HEAD)
run=${ROUTE_A_RUN_DIR:-$SCRATCH/contextgraph_sft/route-a/$commit/$domain/context-$CONTEXT_LENGTH/samples-$SAMPLES-candidates-$CANDIDATES}
mkdir -p "$run"
exec 9>"$run/collection.lock"
flock -n 9 || { echo "Collector already running: $run" >&2; exit 2; }
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} == 4 ]] || { echo 'Expected four nodes' >&2; exit 2; }
export MODEL_PORT=${MODEL_PORT:-18000} SEARCH_PORT=${SEARCH_PORT:-18999}
endpoint=http://${nodes[0]}:$MODEL_PORT
export LOCAL_SEARCH_URL=http://${nodes[1]}:$SEARCH_PORT
# Internal shared server roles accept bcp; this does not select collection data.
export BENCHMARK=bcp SERVER_AUTO_INTERNAL_PORT=1
if [[ "$domain" == scienceworld ]]; then
  unset LOCAL_SEARCH_URL SFT_MANAGED_SEARCH
  python scripts/make_scienceworld_data.py --out-dir "$run/data" --n-train -1 --n-val -1 --seed "$SEED"
  data=$run/data/scienceworld_graph_train.parquet
  heldout=$run/data/scienceworld_graph_val.parquet
  selection=task_balanced
else
  : "${OPENAI_API_KEY:?BC-P requires the configured judge key; no teacher API key is needed}"
  data=${TRAIN_DATA:-$PROJECT_ROOT/data/bc_train.parquet}
  heldout=${HELDOUT_DATA:-$PROJECT_ROOT/data/bc_test.parquet}
  selection=uniform
  export SFT_MANAGED_SEARCH=1
  export SEARCH_HF_HOME=${SEARCH_HF_HOME:-/work/09281/chc_1996/vista/cache}
  export SEARCH_HF_HUB_CACHE=${SEARCH_HF_HUB_CACHE:-$SEARCH_HF_HOME/hub}
fi
collect=(scripts/generate_bcp_graph_sft.py --collection-mode route_a --graph-protocol controller --teacher-profile evaluation --transport vllm_tokens --benchmark "$domain" --data "$data" --heldout "$heldout" --output "$run/collection" --teacher-model "$MODEL_ID" --teacher-checkpoint "$MODEL_PATH" --tokenizer "$MODEL_PATH" --teacher-url "$endpoint" --search-url "${LOCAL_SEARCH_URL:-}" --selection "$selection" --samples "$SAMPLES" --start "${START_INDEX:-0}" --candidates "$CANDIDATES" --workers "$WORKERS" --seed "$SEED" --temperature "${TEMPERATURE:-0.7}" --prompt-length "$PROMPT_LENGTH" --response-length "$((CONTEXT_LENGTH - PROMPT_LENGTH))" --branch-len "${BRANCH_LENGTH:-8192}" --max-turn "${MAX_TURN:-100}" --max-steps "${MAX_STEPS:-100}" --turn-max-tokens "${TURN_MAX_TOKENS:-4096}")
# Validate train split and resume contract before occupying GPUs.
python "${collect[@]}" --dry-run
echo "ROUTE_A_START model=$MODEL_ID thinking=enabled context=$CONTEXT_LENGTH samples=$SAMPLES candidates=$CANDIDATES run=$run"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; wait || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
srun --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" bash scripts/eval_bcp_qwen38_4node_idev.sh _server >"$run/server.log" 2>&1 &
pids+=("$!")
if [[ "$domain" == bcp ]]; then
  srun --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[1]}" bash scripts/eval_bcp_qwen38_4node_idev.sh _search >"$run/search.log" 2>&1 &
  pids+=("$!")
fi
wait_health() {
  local url=$1 elapsed=0
  until curl -fsS --max-time 5 "$url" >/dev/null; do
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null || { echo "Service exited; inspect $run/*.log" >&2; exit 1; }; done
    (( elapsed < ${STARTUP_TIMEOUT:-1800} )) || { echo "Health timeout: $url" >&2; exit 1; }
    echo "Waiting for $url ($elapsed seconds)"
    sleep 30
    elapsed=$((elapsed + 30))
  done
}
wait_health "$endpoint/health"
[[ "$domain" != bcp ]] || wait_health "$LOCAL_SEARCH_URL/health"
srun --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[2]}" "$CONDA_PREFIX/bin/python" -u "${collect[@]}"
export_dir=$(mktemp -d "$run/sft-XXXXXXXX")
build_args=(--require-complete)
[[ "$SAMPLES" == -1 ]] || build_args+=(--smoke)
python scripts/build_route_a_sft.py "$run/collection" --output "$export_dir" --tokenizer "$MODEL_PATH" --max-length "$CONTEXT_LENGTH" --control-ratio "${CONTROL_RATIO:-0.5}" --seed "$SEED" "${build_args[@]}"
echo "ROUTE_A_COMPLETE artifacts=$export_dir (see manifest training_ready and validation_ready)"
