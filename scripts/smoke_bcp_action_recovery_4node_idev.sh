#!/bin/bash
# Focused reproduction of the saved zero-tool cases, not a graph A/B or score.
set -euo pipefail
: "${SCRATCH:?Missing SCRATCH}"
: "${SLURM_JOB_ID:?Run inside the existing four-node idev}"
export EVAL_INDICES_FILE=$(mktemp "$SCRATCH/task-recovery-indices-${SLURM_JOB_ID}-XXXXXX.json")
printf '[28,70]\n' > "$EVAL_INDICES_FILE"
export SAMPLES=-1 GRAPH_RETRY_AB=0 GRAPH_RETRY_GUARD=1 GRAPH_ACTION_PROTOCOL=controller
export EVAL_ACTION_RECOVERY=1 EVAL_MAX_TURN=${EVAL_MAX_TURN:-6}
bash "$(dirname "${BASH_SOURCE[0]}")/smoke_graph_retry_ab_4node_idev.sh"
