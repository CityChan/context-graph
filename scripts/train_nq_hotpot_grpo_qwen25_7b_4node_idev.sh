#!/bin/bash

# Run inside an active four-node Vista idev allocation. Node 0 runs the
# existing BrowseComp-Plus retriever; nodes 1-3 run Qwen2.5-7B GRPO.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
DATA_ROOT=${DATA_ROOT:-${SCRATCH:?SCRATCH must be set}/context-graph-data/searchr1_nq_hotpotqa/processed}
TS=$(date +%Y%m%d_%H%M%S)
RUN_LOG=${RUN_LOG:-$PROJECT_ROOT/logs/train-searchr1-nq-hotpot-qwen25-7b-$TS.log}

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this inside an active four-node idev allocation."
  exit 2
fi

mkdir -p "$PROJECT_ROOT/logs"

export TASK_LABEL="Search-R1 NQ+HotpotQA GRPO diagnostic"
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
export TRAIN_DATA_FILE=${TRAIN_DATA_FILE:-$DATA_ROOT/train.parquet}
export VAL_DATA_FILE=${VAL_DATA_FILE:-$DATA_ROOT/validation_diag.parquet}
export REQUIRE_OPENAI_JUDGE=0
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-12}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TRAIN_LR=${TRAIN_LR:-1e-6}
export PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
export CONTEXT_LENGTH=${CONTEXT_LENGTH:-10240}
export MAX_TURN=${MAX_TURN:-16}
export MAX_SESSION=${MAX_SESSION:-4}
export VAL_MAX_SESSION=${VAL_MAX_SESSION:-4}
export TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-512}
export FINAL_ANSWER_RESERVE=${FINAL_ANSWER_RESERVE:-1024}
export RUN_TAG=${RUN_TAG:-searchr1_nq_hotpot_qwen25_7b_grpo_10step}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-$RUN_TAG-$TS}

set +e
bash "$PROJECT_ROOT/scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh" 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e

echo "Training log: $RUN_LOG"
if [ "$RC" -ne 0 ]; then
  exit "$RC"
fi
python "$PROJECT_ROOT/scripts/audit_skillrl_search_reference.py" --sources searchR1_nq,searchR1_hotpotqa --require-training-health "$RUN_LOG"
