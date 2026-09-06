#!/bin/bash
# Formal evaluator-free GraphRPO specialization. The pre-update policy answers
# paired QA probes from the graph state before and after every valid edit.
set -euo pipefail

export ADV_ESTIMATOR=graphrpo
export POLICY_LOSS_MODE=graphrpo
export BC_CTXGRAPH_PROTOCOL=controller
export PROCESS_REWARD_SPEC='[scope]'
export USE_KL_LOSS=True
export GRAPH_RPO_CREDIT_BACKEND=${GRAPH_RPO_CREDIT_BACKEND:-old_policy_counterfactual_qa}
export GRAPH_RPO_COUNTERFACTUAL_SAMPLES=${GRAPH_RPO_COUNTERFACTUAL_SAMPLES:-2}
export GRAPH_RPO_ALPHA=${GRAPH_RPO_ALPHA:-0.1}
export GRAPH_RPO_DELTA_MAX=${GRAPH_RPO_DELTA_MAX:-0.25}
export RUN_TAG=${RUN_TAG:-graphrpo_5n_48h}

exec bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
