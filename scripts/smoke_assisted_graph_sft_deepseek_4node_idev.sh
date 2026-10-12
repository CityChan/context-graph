#!/bin/bash
# SFT-only assisted demonstrations; task success and evidence gates stay enabled.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run inside an idle existing four-node idev allocation}"
BENCHMARK=${1:-scienceworld}
case "$BENCHMARK" in scienceworld|bcp) ;; *) echo "Usage: $0 [scienceworld|bcp]" >&2; exit 2 ;; esac
# A smoke always gets fresh artifacts; never resume earlier rejected candidates.
unset RUN_DIR ARTIFACT_ROOT RUN_TAG
export RUN_TAG="${SLURM_JOB_ID}-${BENCHMARK}-assisted-$(date +%Y%m%d_%H%M%S)"
export SFT_TEACHER_PROFILE=framework_assisted_v1 AGENT_BUDGET_MODE=working_context
export SAMPLES=${SAMPLES:-3} CANDIDATES=${CANDIDATES:-2} WORKERS=1
export EXPECTED_NUM_NODES=4 TEACHER_TP=4 MAX_NUM_SEQS=1 NUM_WORKERS=1
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.75}
echo "ASSISTED_SFT_SMOKE benchmark=$BENCHMARK questions=$SAMPLES candidates=$CANDIDATES timing=framework decisions=model"
if [[ "$BENCHMARK" == scienceworld ]]; then
    export SFT_SELECTION=task_balanced
    exec bash "$SCRIPT_DIR/generate_scienceworld_graph_sft_deepseek_4node_idev.sh"
fi
export BCP_EXPLICIT_GRAPH_SFT=1 COLOCATE_SEARCH=1
exec bash "$SCRIPT_DIR/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh"
