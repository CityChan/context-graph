#!/bin/bash
# GraphRPO specialization using the trainer's already-loaded frozen reference
# policy to score correct-answer likelihood before and after each graph edit.
set -euo pipefail

export ADV_ESTIMATOR=graphrpo
export POLICY_LOSS_MODE=graphrpo
export BC_CTXGRAPH_PROTOCOL=controller
export PROCESS_REWARD_SPEC='[scope]'
export USE_KL_LOSS=True
export GRAPH_RPO_CREDIT_BACKEND=${GRAPH_RPO_CREDIT_BACKEND:-reference_answer_likelihood}
export RUN_TAG=${RUN_TAG:-graphrpo_5n_48h}

exec bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
