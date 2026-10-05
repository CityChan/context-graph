#!/usr/bin/env bash
# One frozen helper GPU, saved requests only. No actor, corpus server, judge or training.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside the existing idev allocation}"
: "${SCRATCH:?SCRATCH must be set}"
: "${GRAM_HELPER_SOURCE:?Set GRAM_HELPER_SOURCE to the saved GRAM run}"
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export GRAM_HELPER_PYTHON=${GRAM_HELPER_PYTHON:-/work/09281/chc_1996/vista/miniconda3/envs/deepseek_v4/bin/python}
export PYTHONPATH="$PROJECT_ROOT" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MODEL_ID=Qwen/Qwen3.5-9B MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export MODEL_PATH="$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION"
export MODEL_PORT=${MODEL_PORT:-18000}
export SERVER_ENFORCE_EAGER=1 SERVER_COMPACT_JSON=1 SERVER_AUTO_INTERNAL_PORT=1
cd "$PROJECT_ROOT"
[[ -x "$GRAM_HELPER_PYTHON" && -d "$GRAM_HELPER_SOURCE" ]] || { echo 'Missing interpreter or source run'; exit 2; }
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?Missing allocation nodes}")
(( ${#nodes[@]} > 0 )) || exit 2
# The old four-node evaluation's memory node; allow an explicitly selected allocated node.
node=${GRAM_HELPER_NODE:-${nodes[2]:-${nodes[0]}}}
[[ " ${nodes[*]} " == *" $node "* ]] || { echo 'Helper node is outside allocation'; exit 2; }
endpoint="http://$node:$MODEL_PORT"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,$node"
export no_proxy="$NO_PROXY"
mkdir -p "$SCRATCH/context-graph-gram/runs"
run=$(mktemp -d "$SCRATCH/context-graph-gram/runs/helper-probe-${SLURM_JOB_ID}-XXXXXX")
exec > >(tee -a "$run/suite.log") 2>&1
git rev-parse HEAD > "$run/code_commit.txt"
git diff --binary HEAD > "$run/working_tree.patch"
printf '%s\n' "$node" > "$run/node.txt"
echo "GRAM_HELPER_PROBE_START node=$node source=$GRAM_HELPER_SOURCE artifacts=$run"
"$GRAM_HELPER_PYTHON" -c 'import os; from scripts.probe_gram_helper import read_cases; print("Saved helper cases:",len(read_cases(os.environ["GRAM_HELPER_SOURCE"])))'
if curl --connect-timeout 3 --max-time 5 -fsS "$endpoint/health" >/dev/null 2>&1; then
    echo 'Helper endpoint already occupied; refusing to use or stop an existing service'; exit 2
fi
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
srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "$node" env SWE_MODEL_MAX_LEN=65536 WORKERS=1 SEED=42 bash scripts/serve_swe_qwen35_9b_vista.sbatch > "$run/server-memory.log" 2>&1 &
pids+=("$!")
deadline=$((SECONDS + 1200))
until curl --connect-timeout 3 --max-time 5 -fsS "$endpoint/health" >/dev/null 2>&1; do
    kill -0 "${pids[0]}" 2>/dev/null || { tail -n 40 "$run/server-memory.log"; exit 2; }
    (( SECONDS < deadline )) || { echo 'Helper startup timed out'; exit 2; }
    echo 'Waiting for helper only'
    sleep 10
done
srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "$node" "$GRAM_HELPER_PYTHON" -m scripts.probe_gram_helper --source "$GRAM_HELPER_SOURCE" --output "$run" --endpoint "$endpoint" --model "$MODEL_ID" > "$run/probe.log" 2>&1 &
pids+=("$!")
status=0
wait "${pids[1]}" || status=2
if [[ -s "$run/helper-probe.json" ]]; then cat "$run/helper-probe.json"; else tail -n 60 "$run/probe.log"; fi
echo "GRAM_HELPER_PROBE_COMPLETE status=$status artifacts=$run"
exit "$status"
