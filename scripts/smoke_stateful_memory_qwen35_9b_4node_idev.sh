#!/bin/bash
# One native benchmark/method per allocation; defaults to two globally selected tasks.
set -euo pipefail
benchmark=${1:?Expected discoveryworld, scienceworld or swe-lite}
method=${2:?Expected agentfold or supo}
case "$benchmark" in discoveryworld|scienceworld|swe-lite) ;; *) exit 2;; esac
case "$method" in agentfold|supo) ;; *) exit 2;; esac
: "${SLURM_JOB_ID:?Use a four-node idev or the saved sbatch entry}"
: "${SCRATCH:?Missing SCRATCH}"
export PROJECT_ROOT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
cd "$PROJECT_ROOT"
export SAMPLES=${SAMPLES:-2} SERVER_ENFORCE_EAGER=1
[[ "$SAMPLES" == -1 || "$SAMPLES" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid SAMPLES'; exit 2; }
mkdir -p "$PROJECT_ROOT/output"
run=$(mktemp -d "$PROJECT_ROOT/output/$benchmark-$method-${SLURM_JOB_ID}-XXXXXX")
echo "STATEFUL_MEMORY_START benchmark=$benchmark method=$method samples=$SAMPLES artifacts=$run"
rc=0
if [[ "$benchmark" != swe-lite ]]; then
    unset BENCH_DATA BENCH_AGENT_ENV BENCH_RETRY_ERRORS
    export BENCH_RUN_DIR="$run" BENCH_CONTEXT_LENGTH=65536 BENCH_SAMPLES="$SAMPLES"
    export BENCH_BASE_PYTHON=${BENCH_BASE_PYTHON:-/work/09281/chc_1996/vista/miniconda3/envs/cxtgraph/bin/python}
    if [[ "$benchmark" == discoveryworld ]]; then
        export BENCH_MAX_STEPS=200 BENCH_DIFFICULTY=all BENCH_MEMORY_PROFILE=repaired
        export BENCH_DISCOVERYWORLD_OBSERVATION_PROFILE=compact_v1
        bash scripts/eval_discoveryworld_qwen35_9b_4node.sbatch "$method" || rc=1
    else
        export BENCH_MAX_STEPS=100 BENCH_MEMORY_PROFILE=turns BENCH_PROMPT_PROFILE=focus_v2
        export BENCH_REQUIREMENTS=requirements_agent_benchmarks.txt
        unset BENCH_DISCOVERYWORLD_OBSERVATION_PROFILE BENCH_DIFFICULTY
        bash scripts/eval_agent_benchmarks_qwen35_9b_4node.sbatch scienceworld "$method" || rc=1
    fi
    audit_python="$BENCH_BASE_PYTHON"
else
    unset SWE_RETRY_ERRORS SWE_SHARD_INDEX SWE_SHARD_COUNT SWE_ENDPOINT SWE_SERVER_NODE SWE_EVAL_NODE
    export SWE_RUN_DIR="$run" SWE_SAMPLES="$SAMPLES" SWE_CONTEXT_LENGTH=65536
    export SWE_AGENT_ENV=${SWE_AGENT_ENV:-$SCRATCH/context-graph-swe/envs/agent-direct-Rs4ngP}
    bash scripts/eval_swe_lite_arm_4node.sbatch "$method" || rc=1
    audit_python="$SWE_AGENT_ENV/bin/python"
fi
"$audit_python" -I -S "$PROJECT_ROOT/scripts/audit_stateful_memory_smoke.py" "$run" "$benchmark" "$method" --expected "$SAMPLES" || rc=1
echo "STATEFUL_MEMORY_COMPLETE status=$rc artifacts=$run"
exit "$rc"
