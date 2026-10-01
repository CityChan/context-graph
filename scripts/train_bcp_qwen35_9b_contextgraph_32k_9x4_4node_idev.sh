#!/bin/bash
# Use all four nodes in an existing idev allocation; no job submission.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside your existing four-node idev allocation}"
export BCP_TRAIN_PROFILE=contextgraph_32k_three_rank_batch
export BCP_TRAIN_TOPOLOGY=idev4_dp3
export SMOKE_TEST=0 PREFLIGHT_ONLY=0
echo "ContextGraph: 32K / 50 steps / batch 9 x 4 / PPO minibatch 9 (36 global sequence slots)."
echo "Using first node for search and all three remaining nodes for training."
echo "Existing allocation time limit applies; suite.log is captured automatically."
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_bcp_qwen35_9b_50step.sh" contextgraph
