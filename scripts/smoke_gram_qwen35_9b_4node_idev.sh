#!/usr/bin/env bash
# Two real HotpotQA dev questions; no RL and no dataset download on compute nodes.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
: "${SLURM_JOB_ID:?Run inside a four-node idev allocation}"
: "${SCRATCH:?SCRATCH must be set}"
export GRAM_PYTHON=${GRAM_PYTHON:-$SCRATCH/context-graph-swe/envs/agent-direct-Rs4ngP/bin/python}
export GRAM_CONTEXT_LENGTH=${GRAM_CONTEXT_LENGTH:-65536}
export GRAM_MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export MODEL_REVISION=$GRAM_MODEL_REVISION MODEL_ID=Qwen/Qwen3.5-9B
export MODEL_PATH=$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/$GRAM_MODEL_REVISION
export PYTHONPATH="$PROJECT_ROOT" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset QWEN_ENABLE_THINKING
[[ -x "$GRAM_PYTHON" ]] || { echo "Missing evaluator Python: $GRAM_PYTHON" >&2; exit 2; }
case "$GRAM_CONTEXT_LENGTH" in 32768|65536) ;; *) echo 'Expected 32768 or 65536 context' >&2; exit 2;; esac

preload_torch() {
    local library
    library=$("$GRAM_PYTHON" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib/libtorch_global_deps.so")')
    [[ -s "$library" ]] || { echo "Missing Torch global dependencies: $library" >&2; exit 2; }
    export LD_PRELOAD="$library"
}

if [[ ${1:-} == _eval ]]; then
    preload_torch
    exec "$GRAM_PYTHON" -u -m scripts.eval_gram \
      --data "$GRAM_RUN_DIR/data" --output "$GRAM_RUN_DIR/pair-$2" \
      --endpoint "$3" --memory-endpoint "$3" \
      --model "$MODEL_ID" --memory-model "$MODEL_ID" --model-path "$MODEL_PATH" \
      --model-revision "$GRAM_MODEL_REVISION" --memory-revision "$GRAM_MODEL_REVISION" \
      --context-length "$GRAM_CONTEXT_LENGTH" --samples 2 --shard-count 2 --shard-index "$2"
fi
[[ $# == 0 ]] || { echo 'No positional arguments expected' >&2; exit 2; }
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?Missing allocation node list}")
(( ${#nodes[@]} == 4 )) || { echo 'Expected exactly four idev nodes' >&2; exit 2; }
root=$SCRATCH/context-graph-gram
mkdir -p "$root/runs"
export GRAM_RUN_DIR=$(mktemp -d "$root/runs/gram-smoke-9b-${SLURM_JOB_ID}-XXXXXX")
exec 8>"$GRAM_RUN_DIR/launcher.lock"
flock -n 8 || { echo 'Run directory is already locked' >&2; exit 2; }
exec > >(tee -a "$GRAM_RUN_DIR/suite.log") 2>&1
git rev-parse HEAD > "$GRAM_RUN_DIR/code_commit.txt"
git diff --binary HEAD > "$GRAM_RUN_DIR/working_tree.patch"
printf '%s\n' "${nodes[@]}" > "$GRAM_RUN_DIR/nodes.txt"
echo "GRAM_SMOKE_START model=$MODEL_ID context=$GRAM_CONTEXT_LENGTH tasks=2 artifacts=$GRAM_RUN_DIR"
echo 'Protocol: zero-shot; fixed actor/helper; exact-name entity matching; no embedding service.'

# Isolate the TLS preload from the model-server conda environment.
(
    preload_torch
    "$GRAM_PYTHON" -c 'import agents.utils, httpx, filelock, transformers; print("GRAM_EVALUATOR_IMPORT_OK")' || exit 2
    "$GRAM_PYTHON" -m scripts.prepare_gram_data \
      --source examples/gram/hotpotqa_dev_first2.json --benchmark hotpotqa --split validation \
      --output "$GRAM_RUN_DIR/data"
) > "$GRAM_RUN_DIR/preparation.log" 2>&1 || { cat "$GRAM_RUN_DIR/preparation.log"; exit 2; }

pids=()
cleanup() {
    local live pid
    live=" $(jobs -pr | tr '\n' ' ') "
    for pid in "${pids[@]}"; do
        if [[ "$live" == *" $pid "* ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Check both server nodes before launching either service. Server-side GPU locks
# additionally reject occupied GPUs; existing jobs/services are never stopped.
for pair in 0 1; do
    server=${nodes[$((pair * 2))]}
    if curl --connect-timeout 3 --max-time 5 -fsS "http://$server:18000/health" >/dev/null 2>&1; then
        echo "Service already responds on $server:18000; use idle nodes for this smoke"; exit 2
    fi
done
server_pids=()
for pair in 0 1; do
    server=${nodes[$((pair * 2))]}
    echo "GRAM_PAIR shard=$pair/2 server=$server evaluator=${nodes[$((pair * 2 + 1))]}"
    srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "$server" \
      env SWE_MODEL_MAX_LEN="$GRAM_CONTEXT_LENGTH" MODEL_PORT=18000 WORKERS=1 SEED=42 \
      bash scripts/serve_swe_qwen35_9b_vista.sbatch > "$GRAM_RUN_DIR/server-$pair.log" 2>&1 8>&- &
    server_pids+=("$!"); pids+=("$!")
done
deadline=$((SECONDS + 1200))
while true; do
    ready=0
    for pair in 0 1; do
        kill -0 "${server_pids[$pair]}" 2>/dev/null || { echo "Server exited; inspect $GRAM_RUN_DIR/server-$pair.log"; exit 2; }
        if curl --connect-timeout 3 --max-time 5 -fsS "http://${nodes[$((pair * 2))]}:18000/health" >/dev/null 2>&1; then
            ready=$((ready + 1))
        fi
    done
    (( ready == 2 )) && break
    (( SECONDS < deadline )) || { echo 'Server startup timed out'; exit 2; }
    echo "Waiting for servers: $ready/2"
    sleep 15
done
eval_pids=()
for pair in 0 1; do
    srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "${nodes[$((pair * 2 + 1))]}" \
      bash scripts/smoke_gram_qwen35_9b_4node_idev.sh _eval "$pair" "http://${nodes[$((pair * 2))]}:18000" \
      > "$GRAM_RUN_DIR/evaluator-$pair.log" 2>&1 8>&- &
    eval_pids+=("$!"); pids+=("$!")
done
status=0
for pid in "${eval_pids[@]}"; do wait "$pid" || status=2; done
for pair in 0 1; do
    if [[ -s "$GRAM_RUN_DIR/pair-$pair/summary.json" ]]; then cat "$GRAM_RUN_DIR/pair-$pair/summary.json"; fi
done
echo "GRAM_SMOKE_COMPLETE status=$status artifacts=$GRAM_RUN_DIR"
exit "$status"
