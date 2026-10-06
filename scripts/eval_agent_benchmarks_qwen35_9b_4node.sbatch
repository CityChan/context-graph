#!/bin/bash
#SBATCH --job-name=agent-bench-9b
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/agent-bench-9b.%j.out
#SBATCH --error=logs/agent-bench-9b.%j.err

set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
: "${SLURM_JOB_ID:?Use sbatch or a four-node idev allocation}"
: "${SCRATCH:?SCRATCH must be set}"
export BENCH_CONTEXT_LENGTH=${BENCH_CONTEXT_LENGTH:-65536} BENCH_SAMPLES=${BENCH_SAMPLES:--1}
export BENCH_MAX_STEPS=${BENCH_MAX_STEPS:-100}
export BENCH_MEMORY_PROFILE=${BENCH_MEMORY_PROFILE:-repaired}
export BENCH_PROMPT_PROFILE=${BENCH_PROMPT_PROFILE:-legacy}
case "$BENCH_PROMPT_PROFILE" in legacy|focus_v2|discoveryworld_v1) ;; *) echo 'Unknown BENCH_PROMPT_PROFILE'; exit 2 ;; esac
case "$BENCH_MEMORY_PROFILE" in legacy|turns|repaired) ;; *) echo 'Expected BENCH_MEMORY_PROFILE=legacy, turns or repaired'; exit 2 ;; esac
export SERVER_ENFORCE_EAGER=${SERVER_ENFORCE_EAGER:-1}
case "$SERVER_ENFORCE_EAGER" in 0|1) ;; *) echo 'SERVER_ENFORCE_EAGER must be 0 or 1'; exit 2 ;; esac
export MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export MODEL_ID=Qwen/Qwen3.5-9B
export MODEL_PATH=$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION
export BENCH_AGENT_ENV=${BENCH_AGENT_ENV:-$SCRATCH/context-graph-agent-benchmarks/envs/scienceworld-api}
export BENCH_REQUIREMENTS=${BENCH_REQUIREMENTS:-requirements_agent_benchmarks.txt}
export BENCH_DIFFICULTY=${BENCH_DIFFICULTY:-Normal}
export PYTHONPATH="$PROJECT_ROOT" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export STRUCTURED_MEMORY_ENABLED=0
unset QWEN_ENABLE_THINKING
if [[ -n ${JAVA_HOME:-} ]]; then export PATH="$JAVA_HOME/bin:$PATH"; fi

agent_preload() {
    local library
    library=$("$BENCH_AGENT_ENV/bin/python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib/libtorch_global_deps.so")')
    [[ -s "$library" ]] || { echo 'Missing Torch global dependencies'; exit 2; }
    export LD_PRELOAD="$library"
}

if [[ ${1:-} == _eval ]]; then
    agent_preload
    extra=()
    [[ ${BENCH_RETRY_ERRORS:-0} == 1 ]] && extra+=(--retry-errors)
    exec "$BENCH_AGENT_ENV/bin/python" -u scripts/eval_agent_benchmarks.py "$BENCHMARK" --method "$2" --endpoint "$3" --output "$BENCH_RUN_DIR/$4" --shard-index "$5" --shard-count "$6" --data "$BENCH_DATA" --model-path "$MODEL_PATH" --context-length "$BENCH_CONTEXT_LENGTH" --samples "$BENCH_SAMPLES" --max-steps "$BENCH_MAX_STEPS" --memory-profile "$BENCH_MEMORY_PROFILE" --prompt-profile "$BENCH_PROMPT_PROFILE" "${extra[@]}"
fi

export BENCHMARK=${1:?Expected scienceworld}
case "$BENCHMARK" in
    scienceworld) command -v java >/dev/null || { echo 'ScienceWorld requires Java on PATH (or set JAVA_HOME)'; exit 2; } ;;
    discoveryworld) [[ "$BENCH_PROMPT_PROFILE" == discoveryworld_v1 ]] || { echo 'Use the DiscoveryWorld launcher'; exit 2; } ;;
    *) echo 'Expected scienceworld or discoveryworld'; exit 2 ;;
esac
mode=${2:-both}
case "$mode" in
    both) methods=(contextgraph foldagent); labels=(contextgraph foldagent); shard_indices=(0 0); shard_count=1 ;;
    contextgraph|foldagent) methods=("$mode" "$mode"); labels=("$mode-0" "$mode-1"); shard_indices=(0 1); shard_count=2 ;;
    *) echo 'Expected both, contextgraph or foldagent'; exit 2 ;;
esac
case "$BENCH_CONTEXT_LENGTH" in 32768|65536) ;; *) echo 'Expected 32768 or 65536 context'; exit 2;; esac
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#nodes[@]} == 4 )) || { echo 'Expected four nodes (two server/evaluator pairs)'; exit 2; }
root=$SCRATCH/context-graph-agent-benchmarks
export BENCH_DATA=${BENCH_DATA:-$root/data/$BENCHMARK/tasks.json}
mkdir -p "$root/runs"
export BENCH_RUN_DIR=${BENCH_RUN_DIR:-$(mktemp -d "$root/runs/${BENCHMARK}-9b-${mode}-${SLURM_JOB_ID}-XXXXXX")}
mkdir -p "$BENCH_RUN_DIR"
BENCH_RUN_DIR=$(realpath -e "$BENCH_RUN_DIR")
export BENCH_RUN_DIR
exec 8>"$BENCH_RUN_DIR/launcher.lock"
flock -n 8 || { echo 'This run already has a launcher'; exit 2; }
exec > >(tee -a "$BENCH_RUN_DIR/suite.log") 2>&1
echo "BENCHMARK_BATCH_START benchmark=$BENCHMARK method=$mode memory_profile=$BENCH_MEMORY_PROFILE prompt_profile=$BENCH_PROMPT_PROFILE artifacts=$BENCH_RUN_DIR"

# Only this private overlay is modified; preserve cxtgraph and the SWE env.
base=${BENCH_BASE_PYTHON:-/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP/bin/python}
mkdir -p "$(dirname "$BENCH_AGENT_ENV")"
(
    flock -x 7
    requirements_sha=$(sha256sum "$BENCH_REQUIREMENTS" | cut -d' ' -f1)
    if [[ ! -f "$BENCH_AGENT_ENV/.ready" ]]; then
        [[ -x "$base" ]] || { echo "Missing BENCH_BASE_PYTHON: $base"; exit 2; }
        "$base" -m venv --system-site-packages "$BENCH_AGENT_ENV"
        base_site=$("$base" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        overlay_site=$("$BENCH_AGENT_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        printf '%s\n' "$base_site" > "$overlay_site/agent-parent.pth"
        "$BENCH_AGENT_ENV/bin/python" -m pip install -r "$BENCH_REQUIREMENTS"
        "$BENCH_AGENT_ENV/bin/python" -m pip freeze --all > "$BENCH_AGENT_ENV/installed.txt"
        printf '%s\n' "$requirements_sha" > "$BENCH_AGENT_ENV/.ready"
    elif [[ $(cat "$BENCH_AGENT_ENV/.ready") != "$requirements_sha" ]]; then
        echo 'Requirements changed; use a new BENCH_AGENT_ENV to preserve running evaluations'; exit 2
    fi
) 7>"$BENCH_AGENT_ENV.setup.lock"
cp "$BENCH_AGENT_ENV/installed.txt" "$BENCH_RUN_DIR/environment.txt"
if [[ "$BENCHMARK" == discoveryworld ]]; then
    "$BENCH_AGENT_ENV/bin/python" scripts/probe_discoveryworld.py --output "$BENCH_RUN_DIR/simulator-probe" --difficulty "$BENCH_DIFFICULTY"
fi
mkdir -p "$(dirname "$BENCH_DATA")"
(
    flock -x 7
    if [[ ! -s "$BENCH_DATA" ]]; then
        "$BENCH_AGENT_ENV/bin/python" scripts/prepare_agent_benchmarks.py "$BENCHMARK" --output "$(dirname "$BENCH_DATA")" --difficulty "$BENCH_DIFFICULTY"
    fi
) 7>"$BENCH_DATA.prepare.lock"

pids=()
cleanup() {
    local running pid
    running=" $(jobs -pr | tr '\n' ' ') "
    for pid in "${pids[@]}"; do
        if [[ "$running" == *" $pid "* ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
server_pids=()
for pair in 0 1; do
    server=${nodes[$((pair * 2))]}
    if curl --connect-timeout 3 --max-time 5 -fsS "http://$server:18000/health" >/dev/null 2>&1; then
        echo "Existing service on $server:18000; refusing to start a duplicate"; exit 2
    fi
    echo "BENCHMARK_PAIR method=${methods[$pair]} server=$server evaluator=${nodes[$((pair * 2 + 1))]} shard=${shard_indices[$pair]}/$shard_count"
    srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "$server" env SWE_MODEL_MAX_LEN="$BENCH_CONTEXT_LENGTH" MODEL_PORT=18000 WORKERS=1 SEED=42 bash scripts/serve_swe_qwen35_9b_vista.sbatch > "$BENCH_RUN_DIR/server-${labels[$pair]}.log" 2>&1 8>&- &
    server_pids+=("$!"); pids+=("$!")
done
deadline=$((SECONDS + 1200))
while true; do
    ready=0
    for pair in 0 1; do
        kill -0 "${server_pids[$pair]}" 2>/dev/null || { echo 'Server exited; inspect server log'; exit 2; }
        if curl --connect-timeout 3 --max-time 5 -fsS "http://${nodes[$((pair * 2))]}:18000/health" >/dev/null 2>&1; then ready=$((ready + 1)); fi
    done
    (( ready == 2 )) && break
    (( SECONDS < deadline )) || { echo 'Model startup timeout'; exit 2; }
    echo "Waiting for servers: $ready/2"
    sleep 15
done
eval_pids=()
for pair in 0 1; do
    srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "${nodes[$((pair * 2 + 1))]}" bash scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch _eval "${methods[$pair]}" "http://${nodes[$((pair * 2))]}:18000" "${labels[$pair]}" "${shard_indices[$pair]}" "$shard_count" > "$BENCH_RUN_DIR/evaluator-${labels[$pair]}.log" 2>&1 8>&- &
    eval_pids+=("$!"); pids+=("$!")
done
status=0
for pid in "${eval_pids[@]}"; do wait "$pid" || status=2; done
echo "BENCHMARK_BATCH_COMPLETE status=$status artifacts=$BENCH_RUN_DIR"
exit "$status"
