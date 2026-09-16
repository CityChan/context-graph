#!/bin/bash
# Matched zero-shot ContextGraph memory ablation for long-horizon ALFWorld @real.
# Keeps bounded retrieval-history replacement, but removes per-action graph
# dumps and controller checkpoints that an untrained base model cannot use
# reliably. The original full-controller evaluator remains unchanged.
set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this inside a four-node Vista idev allocation"
  exit 1
fi

ALFWORLD_EVAL_SAMPLES=${ALFWORLD_EVAL_SAMPLES:-32}
ALFWORLD_DATA_SEED=${ALFWORLD_DATA_SEED:-42}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}

if ! [[ "$ALFWORLD_EVAL_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: ALFWORLD_EVAL_SAMPLES must be a positive integer"
  exit 1
fi

python scripts/make_alfworld_data.py \
  --mode real \
  --n_train 16 \
  --n_val "$ALFWORLD_EVAL_SAMPLES" \
  --seed "$ALFWORLD_DATA_SEED" \
  --alfworld_data "$ALFWORLD_DATA"

EVAL_TS=$(date +%Y%m%d_%H%M%S)
export ALFWORLD_EXPERIMENT_NAME="contextgraph_memory_alfworld_real_8b_4n_zeroshot_n${ALFWORLD_EVAL_SAMPLES}_seed${ALFWORLD_DATA_SEED}_${EVAL_TS}"
mkdir -p logs
EVAL_LOG="logs/${ALFWORLD_EXPERIMENT_NAME}.log"
EVAL_SUMMARY="logs/${ALFWORLD_EXPERIMENT_NAME}.summary.json"

export WANDB_MODE=disabled
export ALFWORLD_DISABLE_WANDB=1
export MODEL_PATH=Qwen/Qwen3-8B
export ALFWORLD_MODE=real
export ALFWORLD_METHOD_LABEL="ContextGraph memory-only zero-shot"
export ALFWORLD_AGENT_LOOP=context_graph_isolated_agent
export ALFWORLD_WORKFLOW=alfworld_graph
export ALFWORLD_PROCESS_REWARD='[flat,scope,graph]'
export ALFWORLD_VAL_ONLY=True
export ALFWORLD_VAL_BEFORE_TRAIN=True
export ALFWORLD_VAL_MAX_SAMPLES="$ALFWORLD_EVAL_SAMPLES"
export ALFWORLD_TRAIN_MAX_SAMPLES=16
export ALFWORLD_TOTAL_STEPS=1
export ALFWORLD_TRAIN_BATCH_SIZE=4
export ALFWORLD_PPO_MINI_BATCH_SIZE=4
export ALFWORLD_ROLLOUT_N=1
export ALFWORLD_PROMPT_LENGTH=4096
export ALFWORLD_RESPONSE_LENGTH=12288
export ALFWORLD_MAX_TOKEN_LEN_PER_GPU=16384
export ALFWORLD_MAX_TURN=60
export ALFWORLD_VAL_MAX_TURN=60
export ALFWORLD_TURN_MAX_NEW_TOKENS=128
export ALFWORLD_BRANCH_LEN=2048
export ALFWORLD_MAX_SESSION=3
export ALFWORLD_VAL_MAX_SESSION=3
export ALFWORLD_SESSION_TIMEOUT=600
export ALFWORLD_STRUCTURED_GRAPH_CONTROLLER=True
export ALFWORLD_CONTROLLER_OWNED_TOOL_FORMATTING=True
export ALFWORLD_CONTROLLER_ACTION_POLICY=balanced
export ALFWORLD_CONTROLLER_ALLOW_PASS=True
export ALFWORLD_CONSOLIDATION_INTERVAL=0
export ALFWORLD_ENABLE_RETRIEVAL_MEMORY=True
export ALFWORLD_INJECT_GRAPH_STATE_AFTER_ACTION=False
export ALFWORLD_TRAINER_RESUME_MODE=disable

echo "=============================================================="
echo "  ContextGraph memory-only ALFWorld evaluation: ${ALFWORLD_EVAL_SAMPLES} fixed episodes"
echo "  Base model: ${MODEL_PATH}; seed: ${ALFWORLD_DATA_SEED}"
echo "  Raw log: ${EVAL_LOG}"
echo "=============================================================="

set +e
bash scripts/train_alfworld_ctxgraph_8b_4node_30step.sh 2>&1 | tee "$EVAL_LOG"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
  echo "ERROR: ALFWorld evaluation failed with exit code $RC; log: $EVAL_LOG"
  exit "$RC"
fi

python scripts/summarize_alfworld_eval.py "$EVAL_LOG" --samples "$ALFWORLD_EVAL_SAMPLES" --output "$EVAL_SUMMARY"
echo "Evaluation summary: $EVAL_SUMMARY"
