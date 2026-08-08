#!/bin/bash
# Four-node, validation-only FoldAgent smoke on official ScienceAgentBench.
set -eo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"

python tests/smoke_sab_sandbox.py

export SAB_RUN_TAG=formal_fold_smoke
export SAB_REAL_EVAL=1
export SAB_METHOD=fold
export SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:-4}
export SAB_TRAIN_MAX_SAMPLES=${SAB_TRAIN_MAX_SAMPLES:-4}
export SAB_DISABLE_WANDB=1
export SAB_DUMP_VALIDATION=1
export SAB_LOG_VAL_GENERATIONS=${SAB_LOG_VAL_GENERATIONS:-4}
export SAB_DEBUG_IO=${SAB_DEBUG_IO:-1}
export SAB_EXPOSE_EVAL_CONTRACT=0
export SAB_INTERACTIVE_EVAL_FEEDBACK=0
export SAB_PROMPT_LENGTH=${SAB_PROMPT_LENGTH:-16384}
export SAB_RESPONSE_LENGTH=${SAB_RESPONSE_LENGTH:-12288}
export SAB_MAX_TOKEN_LEN_PER_GPU=${SAB_MAX_TOKEN_LEN_PER_GPU:-28672}
export SAB_VAL_MAX_TURN=${SAB_VAL_MAX_TURN:-16}
export SAB_TURN_MAX_NEW_TOKENS=${SAB_TURN_MAX_NEW_TOKENS:-1024}

exec bash scripts/eval_sab_react_8b_4node_smoke.sh
