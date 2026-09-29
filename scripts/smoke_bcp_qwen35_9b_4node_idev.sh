#!/bin/bash
# Use the existing idev allocation; no sbatch and no downloads.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${SLURM_JOB_ID:?Run from the current four-node idev allocation}"
case "${1:-contextgraph}" in
  contextgraph|foldagent) METHODS=("${1:-contextgraph}") ;;
  both) METHODS=(contextgraph foldagent) ;;
  *) echo "Usage: bash $0 [contextgraph|foldagent|both]" >&2; exit 2 ;;
esac
export SMOKE_TEST=1 PREFLIGHT_ONLY=0
for method in "${METHODS[@]}"; do
  bash scripts/train_bcp_qwen35_9b_50step.sh "$method"
done
