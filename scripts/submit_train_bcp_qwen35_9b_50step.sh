#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
METHOD=${1:-both}
case "$METHOD" in
  both) METHODS=(contextgraph foldagent) ;;
  contextgraph|foldagent) METHODS=("$METHOD") ;;
  *) echo "Usage: bash $0 [both|contextgraph|foldagent]" >&2; exit 2 ;;
esac
mkdir -p logs
for method in "${METHODS[@]}"; do
  JOB_ID=$(sbatch --parsable --job-name="bcp-9b-${method}-50" scripts/train_bcp_qwen35_9b_50step.sbatch "$method")
  printf '%s: submitted %s (5 nodes, 48h limit, 50 steps)\n' "$method" "$JOB_ID"
done
