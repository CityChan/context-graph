#!/bin/bash
# Run the remaining FoldAgent benchmark smokes serially in one 4-node idev.
set -uo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"

declare -a STAGES=(gaia sab alfworld)
declare -a SCRIPTS=(
  scripts/smoke_gaia_text_foldagent_api_1node.sh
  scripts/smoke_sab_official_foldagent_8b_4node.sh
  scripts/smoke_alfworld_foldagent_8b_4node.sh
)
declare -a RESULTS=()

for i in "${!STAGES[@]}"; do
  stage=${STAGES[$i]}
  script=${SCRIPTS[$i]}
  echo "=============================================================="
  echo "  START FoldAgent smoke: $stage"
  echo "  Script: $script"
  echo "  Time:   $(date)"
  echo "=============================================================="
  bash "$script"
  rc=$?
  RESULTS+=("$stage=$rc")
  echo "+++ FoldAgent smoke $stage finished with exit $rc"
done

echo "=============================================================="
echo "  FoldAgent serial smoke summary: ${RESULTS[*]}"
echo "=============================================================="

for result in "${RESULTS[@]}"; do
  if [ "${result##*=}" -ne 0 ]; then
    exit 1
  fi
done
exit 0
