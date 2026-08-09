#!/bin/bash
# Four-node BrowseComp-Plus ReAct baseline smoke with one retrieval GPU.
set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"

export BC_DISABLE_WANDB=1
export BC_VAL_MAX_SAMPLES=${BC_VAL_MAX_SAMPLES:-4}
export BC_PROMPT_LENGTH=${BC_PROMPT_LENGTH:-4096}
export BC_RESPONSE_LENGTH=${BC_RESPONSE_LENGTH:-8192}
export BC_MAX_TOKEN_LEN_PER_GPU=${BC_MAX_TOKEN_LEN_PER_GPU:-12288}
export BC_MAX_TURN=${BC_MAX_TURN:-20}
export BC_TURN_MAX_NEW_TOKENS=${BC_TURN_MAX_NEW_TOKENS:-256}
export BC_MAX_SESSION=${BC_MAX_SESSION:-3}
export BC_SEARCH_TIMEOUT_SECONDS=${BC_SEARCH_TIMEOUT_SECONDS:-600}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o-mini}

exec bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh
