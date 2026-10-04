#!/usr/bin/env bash
# Zero-shot GRAM BC-P adaptation: corpus, actor, frozen helper, evaluator.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
: "${SLURM_JOB_ID:?Run inside a four-node idev allocation}"
: "${SCRATCH:?SCRATCH must be set}"
# Match the existing BC-P/DiscoveryBench credential-file workflow. Never log keys.
if [[ -z ${OPENAI_API_KEY:-} || ${OPENAI_API_KEY:-} == dummy ]]; then
    for credentials in "${WORK:-/work/09281/chc_1996/vista}/.openai_env" /work/09281/chc_1996/vista/.openai_env "$HOME/.openai_env"; do
        if [[ -f "$credentials" ]]; then
            source "$credentials"
            if [[ -n ${OPENAI_API_KEY:-} && ${OPENAI_API_KEY:-} != dummy ]]; then break; fi
        fi
    done
fi
: "${OPENAI_API_KEY:?No judge key found in environment or existing .openai_env files}"
[[ "$OPENAI_API_KEY" != dummy ]] || { echo 'A real judge key is required' >&2; exit 2; }
export OPENAI_API_KEY
# The judge must not inherit a local actor's OPENAI_BASE_URL.
if [[ -n ${JUDGE_BASE_URL:-} ]]; then
    export JUDGE_BASE_URL
    export OPENAI_BASE_URL="$JUDGE_BASE_URL"
else
    unset OPENAI_BASE_URL
fi
unset OPENAI_URL
export GRAM_PYTHON=${GRAM_PYTHON:-$SCRATCH/context-graph-swe/envs/agent-direct-Rs4ngP/bin/python}
export GRAM_CONTEXT_LENGTH=${GRAM_CONTEXT_LENGTH:-65536}
export GRAM_SAMPLES=${GRAM_SAMPLES:-2} GRAM_SEED=${GRAM_SEED:-42}
export GRAM_EPISODE_TOKENS=${GRAM_EPISODE_TOKENS:-24576} GRAM_MAX_STEPS=${GRAM_MAX_STEPS:-100}
# Linked worktrees contain tracked code, not the primary checkout's local data.
# Honor explicit DATA_PATH even when missing; never silently replace that choice.
if [[ -z ${DATA_PATH:-} ]]; then
    DATA_PATH="$PROJECT_ROOT/data/bc_test.parquet"
    if [[ ! -s "$DATA_PATH" ]]; then
        common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
        if [[ -n "$common_dir" && -s "$common_dir/../data/bc_test.parquet" ]]; then
            DATA_PATH=$(realpath -e "$common_dir/../data/bc_test.parquet")
        fi
    fi
fi
export DATA_PATH
export MODEL_ID=Qwen/Qwen3.5-9B MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export MODEL_PATH=$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION
export BENCHMARK=bcp JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano}
export PYTHONPATH="$PROJECT_ROOT" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset QWEN_ENABLE_THINKING
[[ -x "$GRAM_PYTHON" ]] || { echo "Missing evaluator Python: $GRAM_PYTHON; set GRAM_PYTHON to an existing compatible interpreter" >&2; exit 2; }
[[ -s "$DATA_PATH" ]] || { echo "Missing or empty BC-P parquet: $DATA_PATH; set DATA_PATH to the existing BC-P test parquet" >&2; exit 2; }
case "$GRAM_CONTEXT_LENGTH" in 32768|65536) ;; *) echo 'Expected 32768 or 65536 context'; exit 2;; esac
preload_torch() {
    local library
    library=$("$GRAM_PYTHON" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib/libtorch_global_deps.so")')
    [[ -s "$library" ]] || { echo "Missing Torch global dependencies: $library" >&2; exit 2; }
    export LD_PRELOAD="$library"
}
if [[ ${1:-} == _eval ]]; then
    preload_torch
    exec "$GRAM_PYTHON" -u -m scripts.eval_gram --benchmark bcp \
      --data "$DATA_PATH" --output "$GRAM_RUN_DIR/evaluation" \
      --endpoint "$GRAM_ACTOR_URL" --memory-endpoint "$GRAM_HELPER_URL" \
      --model "$MODEL_ID" --memory-model "$MODEL_ID" --model-path "$MODEL_PATH" \
      --model-revision "$MODEL_REVISION" --memory-revision "$MODEL_REVISION" \
      --context-length "$GRAM_CONTEXT_LENGTH" --samples "$GRAM_SAMPLES" --seed "$GRAM_SEED" \
      --episode-tokens "$GRAM_EPISODE_TOKENS" --max-steps "$GRAM_MAX_STEPS" --timeout 3600
fi
[[ $# == 0 ]] || { echo 'No positional arguments expected'; exit 2; }
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?Missing allocation nodes}")
(( ${#nodes[@]} == 4 )) || { echo 'Expected exactly four idev nodes'; exit 2; }
mkdir -p "$SCRATCH/context-graph-gram/runs"
export GRAM_RUN_DIR=$(mktemp -d "$SCRATCH/context-graph-gram/runs/gram-bcp-9b-${SLURM_JOB_ID}-XXXXXX")
exec 8>"$GRAM_RUN_DIR/launcher.lock"
flock -n 8 || exit 2
exec > >(tee -a "$GRAM_RUN_DIR/suite.log") 2>&1
git rev-parse HEAD > "$GRAM_RUN_DIR/code_commit.txt"
git diff --binary HEAD > "$GRAM_RUN_DIR/working_tree.patch"
printf '%s\n' "${nodes[@]}" > "$GRAM_RUN_DIR/nodes.txt"
export LOCAL_SEARCH_URL="http://${nodes[0]}:18999"
export GRAM_ACTOR_URL="http://${nodes[1]}:18000" GRAM_HELPER_URL="http://${nodes[2]}:18000"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,${nodes[0]},${nodes[1]},${nodes[2]}"
export no_proxy="$NO_PROXY"
echo "GRAM_BCP_START samples=$GRAM_SAMPLES context=$GRAM_CONTEXT_LENGTH artifacts=$GRAM_RUN_DIR"
echo "Evaluator=$GRAM_PYTHON data=$DATA_PATH"
echo "Nodes: search=${nodes[0]} actor=${nodes[1]} memory=${nodes[2]} evaluator=${nodes[3]}"
echo 'Protocol: zero-shot BC-P adaptation; actor thinking disabled; helper tokens additional; exact-name entity matching.'
(
    preload_torch
    "$GRAM_PYTHON" -c 'import os,agents.utils; from agents.gram_bcp import load_tasks; from agents.gram_agent import GramConfig; tasks,_,indices=load_tasks(os.environ["DATA_PATH"],int(os.environ["GRAM_SAMPLES"]),int(os.environ["GRAM_SEED"])); GramConfig(max_steps=int(os.environ["GRAM_MAX_STEPS"]),max_episode_tokens=int(os.environ["GRAM_EPISODE_TOKENS"])); print("GRAM_BCP_PREFLIGHT_OK",len(tasks),indices)' || exit 2
) > "$GRAM_RUN_DIR/preparation.log" 2>&1 || { cat "$GRAM_RUN_DIR/preparation.log"; exit 2; }
pids=()
cleanup() {
    local live pid
    live=" $(jobs -pr | tr '\n' ' ') "
    for pid in "${pids[@]}"; do
        if [[ "$live" == *" $pid "* ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
urls=("$LOCAL_SEARCH_URL" "$GRAM_ACTOR_URL" "$GRAM_HELPER_URL")
roles=(search actor memory)
for url in "${urls[@]}"; do
    if curl --connect-timeout 3 --max-time 5 -fsS "$url/health" >/dev/null 2>&1; then
        echo "Service already responds at $url; use idle nodes"; exit 2
    fi
done
srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "${nodes[0]}" \
  env SEARCH_PORT=18999 bash scripts/eval_bcp_qwen38_4node_idev.sh _search > "$GRAM_RUN_DIR/search.log" 2>&1 8>&- &
pids+=("$!")
for index in 1 2; do
    srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "${nodes[$index]}" \
      env SWE_MODEL_MAX_LEN="$GRAM_CONTEXT_LENGTH" MODEL_PORT=18000 WORKERS=1 SEED="$GRAM_SEED" \
      bash scripts/serve_swe_qwen35_9b_vista.sbatch > "$GRAM_RUN_DIR/server-${roles[$index]}.log" 2>&1 8>&- &
    pids+=("$!")
done
deadline=$((SECONDS + 1800))
while true; do
    ready=0
    for index in 0 1 2; do
        kill -0 "${pids[$index]}" 2>/dev/null || { echo "${roles[$index]} exited; inspect logs in $GRAM_RUN_DIR"; exit 2; }
        if curl --connect-timeout 3 --max-time 5 -fsS "${urls[$index]}/health" >/dev/null 2>&1; then ready=$((ready+1)); fi
    done
    (( ready == 3 )) && break
    (( SECONDS < deadline )) || { echo 'Service startup timed out'; exit 2; }
    echo "Waiting for services: $ready/3"
    sleep 15
done
srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "${nodes[3]}" \
  bash scripts/smoke_gram_bcp_qwen35_9b_4node_idev.sh _eval > "$GRAM_RUN_DIR/evaluator.log" 2>&1 8>&- &
pids+=("$!")
status=0
wait "${pids[3]}" || status=2
if [[ -s "$GRAM_RUN_DIR/evaluation/summary.json" ]]; then cat "$GRAM_RUN_DIR/evaluation/summary.json"; fi
echo "GRAM_BCP_COMPLETE status=$status artifacts=$GRAM_RUN_DIR"
exit "$status"
