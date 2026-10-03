#!/bin/bash
# Reuse one existing model server and run ARM tasks on the other allocated node.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside the existing two-node idev allocation}"
: "${SCRATCH:?SCRATCH must be set}"
: "${SWE_AGENT_ENV:?Set SWE_AGENT_ENV to the existing dedicated agent environment}"
method=${SWE_METHOD:-contextgraph}
case "$method" in contextgraph|foldagent|react) ;; *) echo 'Invalid SWE_METHOD'; exit 2;; esac
cd "${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}"
mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#nodes[@]} >= 2 )) || { echo 'Requires two allocated nodes: server and evaluator'; exit 2; }
export SWE_SERVER_NODE=${SWE_SERVER_NODE:-${nodes[0]}}
export SWE_ENDPOINT=${SWE_ENDPOINT:-http://${SWE_SERVER_NODE}:18000}
test_node=
for node in "${nodes[@]}"; do
    if [[ "$node" != "$SWE_SERVER_NODE" ]]; then test_node=$node; break; fi
done
test_node=${SWE_EVAL_NODE:-$test_node}
[[ " ${nodes[*]} " == *" $SWE_SERVER_NODE "* && -n "$test_node" ]] || { echo 'Server node must belong to this allocation'; exit 2; }
[[ " ${nodes[*]} " == *" $test_node "* && "$test_node" != "$SWE_SERVER_NODE" ]] || { echo 'Evaluator must be a different allocated node'; exit 2; }
if [[ ${1:-} != _worker ]]; then
    curl --connect-timeout 5 --max-time 15 -fsS "$SWE_ENDPOINT/health"
    exec srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "$test_node" bash "$0" _worker
fi
[[ $(hostname -s) == "$test_node" && $(uname -m) == aarch64 ]] || { echo 'Expected allocated ARM evaluation node'; exit 2; }
set +u
module load tacc-apptainer/1.4.1
set -u
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
root="$(realpath -e "$SCRATCH")/context-graph-swe"
data="$root/data/lite"
agent="$SWE_AGENT_ENV/bin/python"
grade="$SWE_AGENT_ENV/grading/bin/python"
prepare="$SWE_AGENT_ENV/preparation/bin/python"
[[ -x "$agent" && -x "$grade" && -x "$prepare" ]] || { echo 'Missing agent/grading/preparation environments; reuse the successful pilot setup'; exit 2; }
if [[ ! -e "$data" ]]; then
    "$prepare" scripts/eval_swebench_verified.py prepare --dataset lite --revision 6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2 --data-dir "$data"
fi
mkdir -p "$root/runs"
run=${SWE_RUN_DIR:-$(mktemp -d "$root/runs/lite-arm-${method}-${SLURM_JOB_ID}-XXXXXX")}
mkdir -p "$run"
exec > >(tee -a "$run/suite.log") 2>&1
echo "SWE_ARM_SUBSET_START method=$method server=$SWE_SERVER_NODE evaluator=$test_node artifacts=$run"
echo "Resume with SWE_RUN_DIR=$run using the same code, model and context; allocation time limit still applies."
export SWE_AGENT_LD_PRELOAD
SWE_AGENT_LD_PRELOAD=$("$agent" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib/libtorch_global_deps.so")')
[[ -s "$SWE_AGENT_LD_PRELOAD" ]] || { echo 'Missing Torch TLS preload'; exit 2; }
model="$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
extra=()
[[ ${SWE_RETRY_ERRORS:-0} != 1 ]] || extra+=(--retry-errors)
exec "$grade" -u scripts/run_swe_arm_subset.py --method "$method" --shard-index "${SWE_SHARD_INDEX:-0}" --shard-count "${SWE_SHARD_COUNT:-1}" --data-dir "$data" --output "$run" --agent-python "$agent" --model-path "$model" --endpoint "$SWE_ENDPOINT" --context-length "${SWE_CONTEXT_LENGTH:-65536}" --limit "${SWE_SAMPLES:--1}" "${extra[@]}"
