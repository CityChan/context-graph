#!/bin/bash

# Run FoldAgent and ContextGraph one-step training smokes serially inside one
# existing four-node Vista GH idev allocation.

set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"

if [ -z "${SLURM_NODELIST:-}" ]; then
  echo "ERROR: no active SLURM allocation; start a four-node idev session first."
  exit 2
fi

mapfile -t IDEV_NODES < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#IDEV_NODES[@]}" -ne 4 ]; then
  echo "ERROR: expected exactly 4 idev nodes, got ${#IDEV_NODES[@]}."
  exit 2
fi

export RUN_TAG=${RUN_TAG:-idev_4n_smoke_$(date +%Y%m%d_%H%M%S)}
mkdir -p logs
FOLD_LOG="logs/idev-foldagent-${RUN_TAG}.log"
CTXGRAPH_LOG="logs/idev-contextgraph-${RUN_TAG}.log"

echo "RUN_TAG=$RUN_TAG"
echo "nodes=${IDEV_NODES[*]}"
echo "FoldAgent log=$FOLD_LOG"
echo "ContextGraph log=$CTXGRAPH_LOG"

set +e
bash scripts/smoke_train_bc_foldagent_8b_4node_1step_32k_active.sh 2>&1 | tee "$FOLD_LOG"
FOLD_RC=${PIPESTATUS[0]}

bash scripts/smoke_train_bc_ctxgraph_8b_4node_1step_32k_active.sh 2>&1 | tee "$CTXGRAPH_LOG"
CTXGRAPH_RC=${PIPESTATUS[0]}
set -e

echo "FoldAgent exit=$FOLD_RC log=$FOLD_LOG"
echo "ContextGraph exit=$CTXGRAPH_RC log=$CTXGRAPH_LOG"

if [ "$FOLD_RC" -ne 0 ] || [ "$CTXGRAPH_RC" -ne 0 ]; then
  exit 1
fi
