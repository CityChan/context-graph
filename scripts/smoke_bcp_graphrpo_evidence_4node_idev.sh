#!/bin/bash
# Search + three trainer GPUs inside the user's existing four-node allocation.
set -euo pipefail
: "${GRAPH_RPO_TRAIN_DATA:?Set an absolute path to evidence-labelled training parquet}"
export GRAPH_RPO_CREDIT_BACKEND=evidence
export BCP_TRAIN_PROFILE=default BCP_TRAIN_TOPOLOGY=full
export BCP_GRAPH_RPO_SMOKE=0 SMOKE_TEST=1 PREFLIGHT_ONLY=0
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
exec bash "$PROJECT_ROOT/scripts/train_bcp_qwen35_9b_50step.sh" contextgraph
