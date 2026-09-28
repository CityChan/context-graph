#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
METHOD=${1:-both}
case "$METHOD" in
  both) METHODS=(contextgraph foldagent) ;;
  preflight) METHODS=(contextgraph); export PREFLIGHT_ONLY=1 ;;
  contextgraph|foldagent) METHODS=("$METHOD") ;;
  *) echo "Usage: bash $0 [both|contextgraph|foldagent|preflight]" >&2; exit 2 ;;
esac
mkdir -p logs
for method in "${METHODS[@]}"; do
  SBATCH_ARGS=(--parsable --job-name="bcp-9b-${method}-50")
  DESCRIPTION="5 nodes, 48h limit, 50 steps"
  if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
    SBATCH_ARGS=(--parsable --nodes=1 --time=00:10:00 --job-name=bcp-9b-preflight)
    DESCRIPTION="1 node, 10min limit, dependency checks only"
  fi
  JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" scripts/train_bcp_qwen35_9b_50step.sbatch "$method")
  printf '%s: submitted %s (%s)\n' "$method" "$JOB_ID" "$DESCRIPTION"
  printf 'Logs: logs/bcp-9b-rl50.*.%s.out and .err\n' "${JOB_ID%%;*}"
done
