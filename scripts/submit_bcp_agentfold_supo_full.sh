#!/bin/bash
# Submit the two smoke-validated zero-shot adaptations from one pinned checkout.
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PROJECT_ROOT"
command -v sbatch >/dev/null
git diff --quiet HEAD -- agents envs scripts verl || { echo 'Commit tracked code changes before submitting.' >&2; exit 2; }
DATA_SOURCE=${DATA_PATH:-$PROJECT_ROOT/data/bc_test.parquet}
[[ -s "$DATA_SOURCE" ]] || { echo "Missing BC-P data: $DATA_SOURCE" >&2; exit 2; }
OUTPUT_BASE=${OUTPUT_BASE:-$PROJECT_ROOT/output}
mkdir -p "$OUTPUT_BASE"
OUTPUT_BASE=$(cd "$OUTPUT_BASE" && pwd)
SUBMISSION=$(mktemp -d "$OUTPUT_BASE/bcp-agentfold-supo-full-XXXXXX")
REV=$(git rev-parse HEAD)
printf '%s\n' "$REV" > "$SUBMISSION/commit.txt"
git worktree add --detach "$SUBMISSION/code" "$REV"
cp "$DATA_SOURCE" "$SUBMISSION/bc_test.parquet"
sha256sum "$SUBMISSION/bc_test.parquet" > "$SUBMISSION/data.sha256"

# Match the successful smoke settings; inherited smoke paths/counts must not leak.
export PROJECT_ROOT="$SUBMISSION/code" DATA_PATH="$SUBMISSION/bc_test.parquet"
export BENCHMARK=bcp SAMPLES=-1 SEED=42 WORKERS=1 MEMORY_MODE=repaired
export MODEL_MAX_LEN=32768 SERVER_ENFORCE_EAGER=1
export SUPO_CONTEXT_THRESHOLD=16384 SUPO_MAX_SUMMARIES=2 SUPO_SUMMARY_MAX_TOKENS=1024
export AGENT_CONDA_ENV=cxtgraph SERVER_CONDA_ENV=deepseek_v4
unset RUN_ROOT EXPECTED_JOB_ID MODEL_PATH
printf 'method\tjob_id\trun_dir\n' > "$SUBMISSION/jobs.tsv"
echo "BCP_FULL_SUBMISSION commit=$REV artifacts=$SUBMISSION"
for method in agentfold supo; do
  mkdir "$SUBMISSION/$method"
  export RUN_ROOT="$SUBMISSION/$method/run"
  if job=$(sbatch --parsable --export=ALL --nodes=4 --time=24:00:00 \
      --chdir="$PROJECT_ROOT" --output="$SUBMISSION/$method/slurm-%j.out" \
      --error="$SUBMISSION/$method/slurm-%j.err" \
      "$PROJECT_ROOT/scripts/eval_bcp_${method}_qwen35_9b_4node.sbatch"); then
    printf '%s\t%s\t%s\n' "$method" "$job" "$RUN_ROOT" >> "$SUBMISSION/jobs.tsv"
    echo "SUBMITTED method=$method job=$job run=$RUN_ROOT"
  else
    echo "Submission failed for $method; earlier jobs remain active. See $SUBMISSION/jobs.tsv; do not blindly resubmit both." >&2
    exit 1
  fi
done
echo "BCP_FULL_SUBMITTED jobs=$SUBMISSION/jobs.tsv"
