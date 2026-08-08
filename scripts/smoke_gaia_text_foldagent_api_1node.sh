#!/bin/bash
# Single-node GAIA FoldAgent smoke using the local BrowseComp-Plus retriever.
set -eo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"

python -m tests.smoke_gaia_data

export GAIA_WORKFLOW=search_branch
export GAIA_MAX_SAMPLES=${GAIA_MAX_SAMPLES:-4}
export GAIA_NUM_WORKERS=${GAIA_NUM_WORKERS:-1}
export GAIA_MODEL_NAME=${GAIA_MODEL_NAME:-gpt-4o-mini}
export GAIA_TOKENIZER_NAME=${GAIA_TOKENIZER_NAME:-Qwen/Qwen3-8B}
export GAIA_SEARCH_TIMEOUT_SECONDS=${GAIA_SEARCH_TIMEOUT_SECONDS:-600}
export GAIA_PROMPT_LENGTH=${GAIA_PROMPT_LENGTH:-16384}
export GAIA_RESPONSE_LENGTH=${GAIA_RESPONSE_LENGTH:-16384}
export GAIA_MAX_TURN=${GAIA_MAX_TURN:-12}
export GAIA_MAX_SESSION=${GAIA_MAX_SESSION:-3}
export GAIA_BRANCH_LEN=${GAIA_BRANCH_LEN:-8192}
export GAIA_TURN_MAX_NEW_TOKENS=${GAIA_TURN_MAX_NEW_TOKENS:-1024}
export GAIA_SAVE_MESSAGES=${GAIA_SAVE_MESSAGES:-1}

exec bash scripts/eval_gaia_graph_api_smoke.sh
