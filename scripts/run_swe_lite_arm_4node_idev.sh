#!/bin/bash
# Two independent server/evaluator pairs in one existing four-node allocation.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside the four-node idev allocation}"
: "${SCRATCH:?SCRATCH must be set}"
: "${SWE_AGENT_ENV:?Set the existing dedicated SWE agent environment}"
method=${1:-foldagent}
case "$method" in foldagent|contextgraph|react|agentfold|supo) ;; *) echo 'Expected foldagent, contextgraph, react, agentfold or supo'; exit 2;; esac
cd "${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}"
case "$(hostname -s)" in login*) echo 'Run from the idev compute shell'; exit 2;; esac
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#nodes[@]} == 4 )) || { echo 'Expected exactly four allocated nodes'; exit 2; }
context=${SWE_CONTEXT_LENGTH:-65536}
case "$context" in 32768|65536) ;; *) echo 'Expected 32768 or 65536 context'; exit 2;; esac
root="$(realpath -e "$SCRATCH")/context-graph-swe"
mkdir -p "$root/runs"
run=${SWE_RUN_DIR:-$(mktemp -d "$root/runs/lite-arm-${method}-4node-${SLURM_JOB_ID}-XXXXXX")}
mkdir -p "$run"
run=$(realpath -e "$run")
# Prevent two launchers from owning the same pair output directories.
exec 8>"$run/launcher.lock"
flock -n 8 || { echo 'This four-node run already has a launcher'; exit 2; }
exec > >(tee -a "$run/suite.log") 2>&1
echo "SWE_ARM_4NODE_START method=$method context=$context artifacts=$run"
echo "Pairs: server=${nodes[0]} evaluator=${nodes[1]}; server=${nodes[2]} evaluator=${nodes[3]}"
echo "Default: 126 tasks per pair; SWE_SAMPLES limits the global candidate list before splitting."
echo "Resume: set SWE_RUN_DIR=$run with the same method, code and budget."
# Prepare shared metadata once before two evaluators use it.
if [[ ! -e "$root/data/lite" ]]; then
    "$SWE_AGENT_ENV/preparation/bin/python" scripts/eval_swebench_verified.py prepare --dataset lite --revision 6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2 --data-dir "$root/data/lite"
fi
pids=()
server_pids=()
cleanup() {
    local status=$? running pid pair active deadline
    trap - EXIT
    # Only processes started by this launcher; never cancel the allocation.
    running=" $(jobs -pr | tr '\n' ' ') "
    for pid in "${pids[@]}"; do
        if [[ "$running" == *" $pid "* ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
    deadline=$((SECONDS + 60))
    while :; do
        active=0
        running=" $(jobs -pr | tr '\n' ' ') "
        for pid in "${pids[@]}"; do
            [[ "$running" != *" $pid "* ]] || active=1
        done
        # Probe only endpoints we launched, not those rejected by preflight.
        for ((pair=0; pair<${#server_pids[@]}; pair++)); do
            if curl --connect-timeout 1 --max-time 2 -fsS "http://${nodes[$((pair * 2))]}:18000/health" >/dev/null 2>&1; then active=1; fi
        done
        (( active )) || break
        if (( SECONDS >= deadline )); then
            echo 'SERVER_CLEANUP_TIMEOUT: owned jobs/endpoints still active; do not start another benchmark yet.'
            exit 2
        fi
        sleep 1
    done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    echo 'SERVER_CLEANUP_COMPLETE'
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for pair in 0 1; do
    server=${nodes[$((pair * 2))]}
    log="$run/server-$pair.log"
    if curl --connect-timeout 3 --max-time 5 -fsS "http://$server:18000/health" >/dev/null 2>&1; then
        echo "A server already responds on $server:18000; use an unused allocation or the single-pair launcher."
        exit 2
    fi
    echo "Starting server pair=$pair node=$server log=$log"
    srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "$server" env MODEL_PORT=18000 SWE_MODEL_MAX_LEN="$context" WORKERS=1 bash scripts/serve_swe_qwen35_9b_vista.sbatch > "$log" 2>&1 8>&- &
    server_pids+=("$!")
    pids+=("$!")
done
deadline=$((SECONDS + 1200))
while true; do
    ready=0
    for pair in 0 1; do
        if ! kill -0 "${server_pids[$pair]}" 2>/dev/null; then
            echo "Server pair=$pair exited. Inspect $run/server-$pair.log"
            tail -n 30 "$run/server-$pair.log"
            exit 2
        fi
        server=${nodes[$((pair * 2))]}
        if curl --connect-timeout 3 --max-time 5 -fsS "http://$server:18000/health" >/dev/null 2>&1; then ready=$((ready + 1)); fi
    done
    (( ready == 2 )) && break
    (( SECONDS < deadline )) || { echo 'Server readiness timed out; inspect server logs'; exit 2; }
    echo "Waiting for servers: $ready/2 ready"
    sleep 15
done
eval_pids=()
for pair in 0 1; do
    server=${nodes[$((pair * 2))]}
    evaluator=${nodes[$((pair * 2 + 1))]}
    echo "Starting evaluator pair=$pair node=$evaluator log=$run/evaluator-$pair.log"
    env SWE_METHOD="$method" SWE_SERVER_NODE="$server" SWE_EVAL_NODE="$evaluator" SWE_ENDPOINT="http://$server:18000" SWE_CONTEXT_LENGTH="$context" SWE_RUN_DIR="$run/pair-$pair" SWE_SHARD_INDEX="$pair" SWE_SHARD_COUNT=2 bash scripts/run_swe_lite_arm_subset_idev.sh > "$run/evaluator-$pair.log" 2>&1 8>&- &
    eval_pids+=("$!")
    pids+=("$!")
done
status=0
for pair in 0 1; do
    if wait "${eval_pids[$pair]}"; then
        echo "Pair $pair complete"
    else
        rc=$?
        echo "Pair $pair exited $rc; inspect $run/evaluator-$pair.log and pair-$pair/summary.json"
        status=2
    fi
done
echo "SWE_ARM_4NODE_FINISHED status=$status artifacts=$run"
echo "Per-pair results: $run/pair-0/summary.json and $run/pair-1/summary.json (disjoint task sets)."
exit "$status"
