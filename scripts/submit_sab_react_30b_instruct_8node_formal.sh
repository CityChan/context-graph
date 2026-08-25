#!/bin/bash
# Submit one full 102-task Qwen3-30B-A3B-Instruct evaluation on SAB.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"

if [ -n "${WORK:-}" ] && [ -f "$WORK/.openai_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.openai_env"
fi

if [ -n "${OPENAI_API_KEY:-}" ] && [ "${OPENAI_API_KEY:-}" != "dummy" ]; then
  :
elif [ -n "${AZURE_OPENAI_KEY:-}" ] && [ -n "${AZURE_OPENAI_API_VERSION:-}" ] && [ -n "${AZURE_OPENAI_ENDPOINT:-}" ] && [ -n "${AZURE_OPENAI_DEPLOYMENT_NAME:-}" ]; then
  :
else
  echo "ERROR: formal SAB requires OPENAI_API_KEY or the complete Azure OpenAI credential set"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
SAB_METHOD=${SAB_METHOD:-react}
SAB_CTXGRAPH_PROTOCOL=${SAB_CTXGRAPH_PROTOCOL:-legacy}
case "$SAB_METHOD" in
  react|fold|ctxgraph) ;;
  *)
    echo "ERROR: SAB_METHOD must be react, fold, or ctxgraph; got $SAB_METHOD"
    exit 1
    ;;
esac
if [ "$SAB_CTXGRAPH_PROTOCOL" = "controller" ] && [ "$SAB_METHOD" != "ctxgraph" ]; then
  echo "ERROR: SAB_CTXGRAPH_PROTOCOL=controller requires SAB_METHOD=ctxgraph"
  exit 1
fi
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_${SAB_METHOD}_${SAB_CTXGRAPH_PROTOCOL}_sab_30b_instruct_8n_formal_${TS}}

echo "Submitting $EXPERIMENT_NAME: method=$SAB_METHOD protocol=$SAB_CTXGRAPH_PROTOCOL, 8 GH nodes, 102 SAB tasks, 40960-token working context"
sbatch -J "sab30b-${SAB_METHOD}-formal" -N 8 -t 04:00:00 \
  --export=ALL,EXPECTED_NUM_NODES=8,CONDA_ENV_NAME=cxtgraph,MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507,EXPERIMENT_NAME="$EXPERIMENT_NAME",TRAIN_BATCH_SIZE=8,PPO_MINI_BATCH_SIZE=8,SAB_METHOD="$SAB_METHOD",SAB_CTXGRAPH_PROTOCOL="$SAB_CTXGRAPH_PROTOCOL",SAB_RUN_TAG=formal,SAB_REAL_EVAL=1,SAB_DUMP_VALIDATION=1,SAB_VAL_MAX_SAMPLES=-1,SAB_TRAIN_MAX_SAMPLES=8,SAB_DATA_SEED=42,SAB_PROMPT_LENGTH=16384,SAB_RESPONSE_LENGTH=24576,SAB_MAX_TOKEN_LEN_PER_GPU=40960,SAB_VAL_MAX_TURN=32,SAB_TURN_MAX_NEW_TOKENS=2048,SAB_MAX_SESSION=4,SAB_BRANCH_LEN=32768,QWEN_ENABLE_THINKING=False \
  scripts/eval_sab_react_30b_instruct_8node_smoke.sh
