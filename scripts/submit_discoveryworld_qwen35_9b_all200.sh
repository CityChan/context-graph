#!/bin/bash
# Submit the complete DiscoveryWorld suite for both methods from a Vista login node.
set -euo pipefail
[[ $# == 0 ]] || { echo "Usage: bash $0" >&2; exit 2; }
: "${SCRATCH:?SCRATCH must be set}"
command -v sbatch >/dev/null
source_root=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
[[ -z $(git -C "$source_root" status --porcelain --untracked-files=no) ]] || {
    echo 'Commit tracked changes before submitting a reproducible evaluation.' >&2; exit 2;
}
revision=$(git -C "$source_root" rev-parse HEAD)
benchmark_root="$SCRATCH/context-graph-agent-benchmarks"
mkdir -p "$benchmark_root/submissions" "$benchmark_root/runs"
submission=$(mktemp -d "$benchmark_root/submissions/discoveryworld-all200-XXXXXX")
code="$submission/code"
git -C "$source_root" worktree add --detach "$code" "$revision"
mkdir -p "$submission/logs"
printf '%s\n' "$revision" > "$submission/commit.txt"
printf 'method\tjob_id\trun_dir\tstdout\tstderr\n' > "$submission/jobs.tsv"
printf 'DISCOVERYWORLD_SUBMISSION path=%s commit=%s\n' "$submission" "$revision"

# Fixed paired protocol: 8 scenarios x 3 difficulties x 5 seeds per method.
# Ignore stale BENCH_* paths from prior interactive runs. Only private overlays
# are installed by the evaluator; existing cxtgraph/deepseek environments stay intact.
for method in contextgraph foldagent; do
    run_dir=$(mktemp -d "$benchmark_root/runs/discoveryworld-9b-$method-all200-XXXXXX")
    stdout="$submission/logs/$method-%j.out"
    stderr="$submission/logs/$method-%j.err"
    if ! reply=$(env -u BENCH_DATA -u BENCH_AGENT_ENV -u BENCH_RETRY_ERRORS \
        PROJECT_ROOT="$code" BENCH_RUN_DIR="$run_dir" \
        BENCH_BASE_PYTHON=/work/09281/chc_1996/vista/miniconda3/envs/cxtgraph/bin/python \
        BENCH_DISCOVERYWORLD_OBSERVATION_PROFILE=compact_v1 \
        BENCH_CONTEXT_LENGTH=65536 BENCH_SAMPLES=-1 BENCH_MAX_STEPS=200 \
        BENCH_DIFFICULTY=all BENCH_MEMORY_PROFILE=repaired SERVER_ENFORCE_EAGER=1 \
        sbatch --parsable --export=ALL --job-name="dw-$method-all200" \
        --partition=gh --account=AST24021 --nodes=4 --ntasks-per-node=1 \
        --cpus-per-task=72 --time=24:00:00 --chdir="$code" \
        --output="$stdout" --error="$stderr" \
        "$code/scripts/eval_discoveryworld_qwen35_9b_4node.sbatch" "$method"); then
        printf 'Submission failed for %s. Previously submitted jobs are retained; inspect %s/jobs.tsv before retrying.\n' "$method" "$submission" >&2
        exit 1
    fi
    job_id=${reply%%;*}
    printf '%s\t%s\t%s\t%s\t%s\n' "$method" "$job_id" "$run_dir" "$stdout" "$stderr" >> "$submission/jobs.tsv"
    printf 'SUBMITTED method=%s job_id=%s artifacts=%s stdout=%s stderr=%s\n' "$method" "$job_id" "$run_dir" "$stdout" "$stderr"
done
printf 'Submission record: %s/jobs.tsv\n' "$submission"
