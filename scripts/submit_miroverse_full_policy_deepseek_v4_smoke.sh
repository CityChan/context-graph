#!/bin/bash
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
DATA_PATH=${DATA_PATH:-$SCRATCH/contextgraph_sft/miroverse_full_policy/seeds.parquet}
MAX_SAMPLES=${MAX_SAMPLES:-20}
START_INDEX=${START_INDEX:-0}
RUN_TAG=${RUN_TAG:-miroverse-full-policy-smoke}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/miroverse_full_policy_native/$RUN_TAG}

test -s "$DATA_PATH"
cd "$PROJECT_ROOT"
sbatch --export=ALL,DATA_PATH="$DATA_PATH",MAX_SAMPLES="$MAX_SAMPLES",START_INDEX="$START_INDEX",RUN_TAG="$RUN_TAG",ARTIFACT_ROOT="$ARTIFACT_ROOT",FULL_POLICY_CURATOR=1 scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh
