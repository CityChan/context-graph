#!/bin/bash
# A separate collection profile; existing Route A defaults remain unchanged.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${SLURM_JOB_ID:?Run inside an idle four-node idev allocation}"
: "${SCRATCH:?Missing SCRATCH}"
: "${DISTILL_TEACHER_CHECKPOINT:?Set the local exported HF teacher checkpoint}"
export SFT_COLLECTION_MODE=explicit_distillation
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
export ROUTE_A_RUN_DIR=${DISTILL_RUN_DIR:-$SCRATCH/contextgraph_sft/explicit-distillation/$SLURM_JOB_ID-$(date +%Y%m%d_%H%M%S)}
exec bash "$SCRIPT_DIR/generate_route_a_sft_qwen35_9b_4node_idev.sh" "${1:-bcp}"
