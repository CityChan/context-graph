#!/bin/bash
# Only retry invalid_tool_limit tasks; keep original row numbers and RNG seeds.
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: bash $0 ORIGINAL_AGENTFOLD_RUN" >&2; exit 2; }
: "${SLURM_JOB_ID:?Run inside a four-node idev allocation}"
export PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_run=$(cd "$1" && pwd)
export CONDA_SH=${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}
export AGENT_CONDA_ENV=cxtgraph
set +u
source "$CONDA_SH"
conda activate "$AGENT_CONDA_ENV"
set -u
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/output"
retry_root=$(mktemp -d "$PROJECT_ROOT/output/bcp-agentfold-retry-${SLURM_JOB_ID}-XXXXXX")
python scripts/prepare_agentfold_retry.py "$source_run" "$retry_root"
source "$retry_root/retry-env.sh"
unset EXPECTED_JOB_ID
rc=0
# Use the common launcher directly to preserve the original checkpoint path.
bash scripts/eval_bcp_qwen38_4node_idev.sh agentfold || rc=1
audit_ready=true
for rank in 0 1 2; do
  [[ -s "$RUN_ROOT/manifest-$rank.json" && -f "$RUN_ROOT/results-$rank.jsonl" ]] || audit_ready=false
done
if [[ "$audit_ready" == true ]]; then
  python scripts/audit_agentfold_smoke.py "$RUN_ROOT" || rc=1
else
  echo "RETRY_AUDIT_SKIPPED: incomplete evaluator artifacts; inspect $RUN_ROOT"
  rc=1
fi
echo "AGENTFOLD_FAILED_RETRY_COMPLETE status=$rc plan=$retry_root/retry-plan.json artifacts=$RUN_ROOT"
exit "$rc"
