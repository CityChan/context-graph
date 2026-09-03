#!/bin/bash
# GraphRPO specialization of the existing BrowseComp ContextGraph launcher.
# The frozen evaluator service must implement contextgraph.graph_evaluator.v1.
set -euo pipefail

: "${GRAPH_RPO_EVALUATOR_URL:?Set GRAPH_RPO_EVALUATOR_URL to the frozen evaluator /score endpoint}"

export ADV_ESTIMATOR=graphrpo
export POLICY_LOSS_MODE=graphrpo
export BC_CTXGRAPH_PROTOCOL=controller
export PROCESS_REWARD_SPEC='[scope]'
export USE_KL_LOSS=True
export RUN_TAG=${RUN_TAG:-graphrpo_5n_48h}

exec bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
