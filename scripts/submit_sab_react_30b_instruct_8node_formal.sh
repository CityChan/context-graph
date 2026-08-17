#!/bin/bash
# Submit the full 102-task Qwen3-30B-A3B-Instruct ReAct evaluation on SAB.

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
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_react_sab_30b_instruct_8n_formal_${TS}}

echo "Submitting $EXPERIMENT_NAME: 8 GH nodes, 102 SAB tasks, 40960-token working context"
sbatch -J sab30b-react-formal -N 8 -t 04:00:00 \
  --export=ALL,EXPECTED_NUM_NODES=8,CONDA_ENV_NAME=cxtgraph,MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507,EXPERIMENT_NAME="$EXPERIMENT_NAME",TRAIN_BATCH_SIZE=8,PPO_MINI_BATCH_SIZE=8,SAB_RUN_TAG=formal,SAB_REAL_EVAL=1,SAB_DUMP_VALIDATION=1,SAB_VAL_MAX_SAMPLES=-1,SAB_TRAIN_MAX_SAMPLES=8,SAB_PROMPT_LENGTH=16384,SAB_RESPONSE_LENGTH=24576,SAB_MAX_TOKEN_LEN_PER_GPU=40960,SAB_VAL_MAX_TURN=32,SAB_TURN_MAX_NEW_TOKENS=2048,QWEN_ENABLE_THINKING=False \
  scripts/eval_sab_react_30b_instruct_8node_smoke.sh
