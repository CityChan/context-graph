#!/bin/bash
# Validate real summary/resume/finish before launching a full evaluation.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
: "${SLURM_JOB_ID:?Run inside an existing four-node idev}"
: "${SCRATCH:?Missing Vista scratch directory}"
export SAMPLES=${SAMPLES:-3} WORKERS=${WORKERS:-1} BENCHMARK=bcp MEMORY_MODE=repaired
if [[ -z ${RUN_ROOT:-} ]]; then
  SMOKE_PARENT=$(mktemp -d "$SCRATCH/bcp-supo-smoke-${SLURM_JOB_ID}-XXXXXX")
  RUN_ROOT="$SMOKE_PARENT/run"
fi
export RUN_ROOT
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || { echo "Run directory already exists: $RUN_ROOT; choose a new RUN_ROOT"; exit 2; }
export CONDA_SH=${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}
export AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
set +u
source "$CONDA_SH"
conda activate "$AGENT_CONDA_ENV"
set -u
cd "$PROJECT_ROOT"
rc=0
bash scripts/eval_bcp_qwen35_9b_4node_idev.sh supo || rc=1
audit_ready=true
for rank in 0 1 2; do
  [[ -s "$RUN_ROOT/manifest-$rank.json" && -f "$RUN_ROOT/results-$rank.jsonl" ]] || audit_ready=false
done
if [[ "$audit_ready" == true ]]; then
  python scripts/audit_supo_smoke.py "$RUN_ROOT" || rc=1
else
  echo "SUPO_SMOKE_AUDIT_SKIPPED: evaluator artifacts are incomplete; inspect launcher output and $RUN_ROOT"
  rc=1
fi
echo "SUPO_SMOKE_COMPLETE status=$rc artifacts=$RUN_ROOT"
exit "$rc"
