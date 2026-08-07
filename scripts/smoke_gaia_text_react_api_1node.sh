#!/bin/bash
# Single-node end-to-end smoke for the text-only GAIA integration.
set -eo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"

# Validate GAIA filtering and per-workflow parquet construction first.
python -m tests.smoke_gaia_data

# ReAct isolates benchmark/search/reward wiring from branch and graph policy.
export GAIA_WORKFLOW=search
export GAIA_MAX_SAMPLES=4
export GAIA_NUM_WORKERS=1
export GAIA_MODEL_NAME=gpt-4o-mini
export GAIA_TOKENIZER_NAME=Qwen/Qwen3-8B
export GAIA_SEARCH_TIMEOUT_SECONDS=600
export GAIA_PROMPT_LENGTH=16384
export GAIA_RESPONSE_LENGTH=16384
export GAIA_MAX_TURN=12
export GAIA_TURN_MAX_NEW_TOKENS=1024
export GAIA_SAVE_MESSAGES=1

exec bash scripts/eval_gaia_graph_api_smoke.sh
