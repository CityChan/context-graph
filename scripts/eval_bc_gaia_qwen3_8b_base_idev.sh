#!/bin/bash

# Run matched Qwen3-8B ContextGraph evaluations on an
# existing four- or five-node Vista idev allocation. BC-P uses four nodes;
# GAIA uses the complete allocation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/09281/chc_1996}
EVAL_VARIANT=${EVAL_VARIANT:-base}
EVAL_MODEL_TAG=${EVAL_MODEL_TAG:-qwen3_8b_base}
EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-${BASE_EVAL_MODEL_PATH:-Qwen/Qwen3-8B}}
EVAL_CTXGRAPH_PROTOCOL=${EVAL_CTXGRAPH_PROTOCOL:-controller}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_BC=${RUN_BC:-1}
RUN_GAIA=${RUN_GAIA:-1}
BC_CONTEXT_LENGTH=${BC_CONTEXT_LENGTH:-65536}
BC_PROMPT_LENGTH=${BC_PROMPT_LENGTH:-8192}
BC_RESPONSE_LENGTH=${BC_RESPONSE_LENGTH:-$((BC_CONTEXT_LENGTH - BC_PROMPT_LENGTH))}
BC_MAX_TURN=${BC_MAX_TURN:-100}
BC_MAX_SESSION=${BC_MAX_SESSION:-10}
BC_FINAL_ANSWER_RESERVE=${BC_FINAL_ANSWER_RESERVE:-1024}
BC_CONSOLIDATION_INTERVAL=${BC_CONSOLIDATION_INTERVAL:-5}
BC_AUTO_PRUNE_MAX_ACTIVE=${BC_AUTO_PRUNE_MAX_ACTIVE:-12}
GAIA_CONTEXT_LENGTH=${GAIA_CONTEXT_LENGTH:-32768}
GAIA_PROMPT_LENGTH=${GAIA_PROMPT_LENGTH:-8192}
GAIA_RESPONSE_LENGTH=${GAIA_RESPONSE_LENGTH:-$((GAIA_CONTEXT_LENGTH - GAIA_PROMPT_LENGTH))}
GAIA_MAX_TURN=${GAIA_MAX_TURN:-100}
GAIA_MAX_SESSION=${GAIA_MAX_SESSION:-10}
GAIA_FINAL_ANSWER_RESERVE=${GAIA_FINAL_ANSWER_RESERVE:-1024}
GAIA_CONSOLIDATION_INTERVAL=${GAIA_CONSOLIDATION_INTERVAL:-5}
GAIA_AUTO_PRUNE_MAX_ACTIVE=${GAIA_AUTO_PRUNE_MAX_ACTIVE:-12}

case "$RUN_BC:$RUN_GAIA" in
  0:0|0:1|1:0|1:1) ;;
  *) echo "ERROR: RUN_BC and RUN_GAIA must each be 0 or 1" >&2; exit 2 ;;
esac
case "$EVAL_VARIANT:$EVAL_MODEL_TAG" in
  *[!A-Za-z0-9_:-]*) echo "ERROR: EVAL_VARIANT and EVAL_MODEL_TAG may contain only letters, numbers, underscores, and hyphens" >&2; exit 2 ;;
esac
case "$EVAL_CTXGRAPH_PROTOCOL" in
  legacy|full_policy|controller) ;;
  *) echo "ERROR: EVAL_CTXGRAPH_PROTOCOL must be legacy, full_policy, or controller" >&2; exit 2 ;;
esac

if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this script inside an active multi-node idev allocation" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NUM_ALLOC_NODES=${#ALLOC_NODES[@]}
if [ "$NUM_ALLOC_NODES" -lt 4 ] || [ "$NUM_ALLOC_NODES" -gt 5 ]; then
  echo "ERROR: expected a four- or five-node idev allocation, got $NUM_ALLOC_NODES nodes" >&2
  exit 3
fi

FULL_NODELIST=$SLURM_JOB_NODELIST
BC_NODELIST=$(printf '%s\n' "${ALLOC_NODES[@]:0:4}" | paste -sd, -)
BC_EXPERIMENT="eval_bc_${EVAL_VARIANT}_${EVAL_CTXGRAPH_PROTOCOL}_${EVAL_MODEL_TAG}_4n_${BC_CONTEXT_LENGTH}ctx_${STAMP}"
GAIA_EXPERIMENT="eval_gaia_${EVAL_VARIANT}_${EVAL_CTXGRAPH_PROTOCOL}_${EVAL_MODEL_TAG}_${NUM_ALLOC_NODES}n_${GAIA_CONTEXT_LENGTH}ctx_${STAMP}"
GAIA_OUTPUT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$GAIA_EXPERIMENT"
if [ "$NUM_ALLOC_NODES" -eq 4 ]; then
  GAIA_TRAIN_BATCH_SIZE=30
  GAIA_PPO_MINI_BATCH_SIZE=15
else
  GAIA_TRAIN_BATCH_SIZE=32
  GAIA_PPO_MINI_BATCH_SIZE=16
fi

echo "=============================================================="
echo "  Sequential BC-P + GAIA ContextGraph evaluation"
echo "  Job: $SLURM_JOB_ID"
echo "  Variant: $EVAL_VARIANT"
echo "  Model: $EVAL_MODEL_PATH"
echo "  Protocol: $EVAL_CTXGRAPH_PROTOCOL"
echo "  BC-P nodes: $BC_NODELIST"
echo "  GAIA nodes: $FULL_NODELIST ($NUM_ALLOC_NODES nodes)"
echo "  Stages: BC-P=$RUN_BC GAIA=$RUN_GAIA"
echo "  BC-P budget: context=$BC_CONTEXT_LENGTH turns=$BC_MAX_TURN consolidation=$BC_CONSOLIDATION_INTERVAL auto_prune=$BC_AUTO_PRUNE_MAX_ACTIVE"
echo "  GAIA budget: context=$GAIA_CONTEXT_LENGTH turns=$GAIA_MAX_TURN consolidation=$GAIA_CONSOLIDATION_INTERVAL auto_prune=$GAIA_AUTO_PRUNE_MAX_ACTIVE"
echo "=============================================================="

set +e
BC_RC=0
if [ "$RUN_BC" = "1" ]; then
  echo "[1/2] Starting BC-P evaluation"
  (
    export SLURM_JOB_NODELIST="$BC_NODELIST"
    export EXPECTED_NUM_NODES=4
    export MODEL_PATH="$EVAL_MODEL_PATH"
    export HF_HOME HF_HUB_CACHE
    export BC_METHOD=contextgraph
    export BC_CTXGRAPH_PROTOCOL="$EVAL_CTXGRAPH_PROTOCOL"
    export BC_CONTROLLER_ACTION_POLICY=structural
    export BC_EXPERIMENT_MODEL_TAG="$EVAL_MODEL_TAG"
    export BC_CONTEXT_LENGTH BC_PROMPT_LENGTH BC_RESPONSE_LENGTH
    export BC_MAX_TURN BC_MAX_SESSION BC_FINAL_ANSWER_RESERVE
    export BC_CONSOLIDATION_INTERVAL BC_AUTO_PRUNE_MAX_ACTIVE
    export BC_YARN_FACTOR=2.0
    export BC_YARN_ORIGINAL_LENGTH=32768
    export BC_VAL_MAX_SAMPLES=-1
    export BC_DISABLE_WANDB=1
    export WANDB_MODE=disabled
    export EXPERIMENT_NAME="$BC_EXPERIMENT"
    bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh
  )
  BC_RC=$?
  echo "[1/2] BC-P finished with exit code $BC_RC"
else
  echo "[1/2] Skipping BC-P because RUN_BC=0"
fi

GAIA_RC=0
if [ "$RUN_GAIA" = "1" ]; then
  echo "[2/2] Starting GAIA evaluation"
  (
    export SLURM_JOB_NODELIST="$FULL_NODELIST"
    export EXPECTED_NUM_NODES="$NUM_ALLOC_NODES"
    export MODEL_PATH="$EVAL_MODEL_PATH"
    export HF_HOME HF_HUB_CACHE
    export EXPERIMENT_NAME="$GAIA_EXPERIMENT"
    export CHECKPOINT_ROOT="$GAIA_OUTPUT_ROOT"
    export TRAIN_DATA_FILE=data/gaia_validation_graph.parquet
    export VAL_DATA_FILE=data/gaia_validation_graph.parquet
    export TRAIN_MAX_SAMPLES=-1
    export VAL_MAX_SAMPLES=-1
    export TRAINER_VAL_ONLY=True
    export VAL_BEFORE_TRAIN=True
    export TOTAL_TRAINING_STEPS=1
    export TEST_FREQ=999
    export SAVE_FREQ=-1
    export PROMPT_LENGTH="$GAIA_PROMPT_LENGTH"
    export RESPONSE_LENGTH="$GAIA_RESPONSE_LENGTH"
    export CONTEXT_LENGTH="$GAIA_CONTEXT_LENGTH"
    export TRAIN_BATCH_SIZE="$GAIA_TRAIN_BATCH_SIZE"
    export PPO_MINI_BATCH_SIZE="$GAIA_PPO_MINI_BATCH_SIZE"
    export ROLLOUT_N=1
    export MAX_TURN="$GAIA_MAX_TURN"
    export MAX_SESSION="$GAIA_MAX_SESSION"
    export VAL_MAX_SESSION="$GAIA_MAX_SESSION"
    export TURN_MAX_NEW_TOKENS=2048
    export FINAL_ANSWER_RESERVE="$GAIA_FINAL_ANSWER_RESERVE"
    export CONSOLIDATION_INTERVAL="$GAIA_CONSOLIDATION_INTERVAL"
    export AUTO_PRUNE_MAX_ACTIVE="$GAIA_AUTO_PRUNE_MAX_ACTIVE"
    export SESSION_TIMEOUT=3600
    export BC_SEARCH_TIMEOUT_SECONDS=600
    export BC_CTXGRAPH_PROTOCOL="$EVAL_CTXGRAPH_PROTOCOL"
    export BC_CONTROLLER_ACTION_POLICY=balanced
    export BC_DISABLE_WANDB=1
    export WANDB_MODE=disabled
    bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
  )
  GAIA_RC=$?
  echo "[2/2] GAIA finished with exit code $GAIA_RC"
else
  echo "[2/2] Skipping GAIA because RUN_GAIA=0"
fi
set -e

echo "BC-P experiment: $BC_EXPERIMENT"
echo "GAIA experiment: $GAIA_EXPERIMENT"

if [ "$BC_RC" -ne 0 ] || [ "$GAIA_RC" -ne 0 ]; then
  echo "ERROR: one or more evaluations failed: BC-P=$BC_RC GAIA=$GAIA_RC" >&2
  exit 4
fi
