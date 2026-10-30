#!/bin/bash
# Explicit opt-in. 64K/four-node/one-step smoke; SMOKE_TEST=0 selects five nodes.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export GRAPH_RPO_GUARD_PROFILE=prune_search_v1
exec bash "$SCRIPT_DIR/train_bcp_qwen35_9b_guarded_rl.sh"
