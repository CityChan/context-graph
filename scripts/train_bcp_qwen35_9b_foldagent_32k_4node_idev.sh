#!/bin/bash
# Run directly inside an existing four-node idev allocation; no job submission.
set -euo pipefail
: "${SLURM_JOB_ID:?Run inside your existing four-node idev allocation}"
export BCP_TRAIN_PROFILE=foldagent_32k_paper_batch
export BCP_TRAIN_TOPOLOGY=idev4_dp2
export SMOKE_TEST=0 PREFLIGHT_ONLY=0
echo "FoldAgent: 32K / 50 steps / batch 32 x 8 / PPO minibatch 128."
echo "Using first node for search, next two for training; fourth node stays unused."
echo "Existing allocation time limit applies; suite.log is captured automatically."
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_bcp_qwen35_9b_50step.sh" foldagent
