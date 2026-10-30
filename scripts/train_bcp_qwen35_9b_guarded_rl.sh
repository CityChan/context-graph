#!/bin/bash
# Default: four-node, one-step 64K smoke. SMOKE_TEST=0 selects the existing five-node full recipe.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
export GRAPH_RPO_GUARD_PROFILE=${GRAPH_RPO_GUARD_PROFILE:-graph_management_full_v1}
case "$GRAPH_RPO_GUARD_PROFILE" in
  graph_management_v1|graph_management_full_v1) export GRAPH_RPO_CREDIT_BACKEND=management ;;
  invalid_action_v1|prune_search_v1) export GRAPH_RPO_CREDIT_BACKEND=outcome_only ;;
  *) echo 'Unknown guarded RL profile' >&2; exit 2 ;;
esac
export GRAPH_RPO_ACTION_PROTOCOL=explicit_tools
export SMOKE_TEST=${SMOKE_TEST:-1}
export BCP_GRAPH_RPO_SMOKE=0 BCP_TRAIN_TOPOLOGY=full
if [ "$SMOKE_TEST" = 1 ]; then
  export BCP_TRAIN_PROFILE=default
else
  export BCP_TRAIN_PROFILE=contextgraph_64k_paper_batch
fi
exec bash "$SCRIPT_DIR/train_bcp_qwen35_9b_50step.sh" contextgraph
