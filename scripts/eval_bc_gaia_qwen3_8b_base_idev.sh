#!/bin/bash

# Run matched zero-shot Qwen3-8B ContextGraph-controller evaluations on an
# existing four- or five-node Vista idev allocation. BC-P uses four nodes;
# GAIA uses the complete allocation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/09281/chc_1996}
BASE_MODEL_PATH=${BASE_EVAL_MODEL_PATH:-Qwen/Qwen3-8B}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

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
BC_EXPERIMENT="eval_bc_base_controller_qwen3_8b_4n_64k_${STAMP}"
GAIA_EXPERIMENT="eval_gaia_base_controller_qwen3_8b_${NUM_ALLOC_NODES}n_32k_${STAMP}"
GAIA_OUTPUT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$GAIA_EXPERIMENT"

echo "=============================================================="
echo "  Sequential BC-P + GAIA base-model controller evaluation"
echo "  Job: $SLURM_JOB_ID"
echo "  Model: $BASE_MODEL_PATH"
echo "  BC-P nodes: $BC_NODELIST"
echo "  GAIA nodes: $FULL_NODELIST ($NUM_ALLOC_NODES nodes)"
echo "=============================================================="

echo "[1/2] Starting BC-P base-model evaluation"
set +e
(
  export SLURM_JOB_NODELIST="$BC_NODELIST"
  export EXPECTED_NUM_NODES=4
  export MODEL_PATH="$BASE_MODEL_PATH"
  export HF_HOME HF_HUB_CACHE
  export BC_METHOD=contextgraph
  export BC_CTXGRAPH_PROTOCOL=controller
  export BC_CONTROLLER_ACTION_POLICY=structural
  export BC_EXPERIMENT_MODEL_TAG=qwen3_8b_base
  export BC_CONTEXT_LENGTH=65536
  export BC_PROMPT_LENGTH=8192
  export BC_RESPONSE_LENGTH=57344
  export BC_YARN_FACTOR=2.0
  export BC_YARN_ORIGINAL_LENGTH=32768
  export BC_FINAL_ANSWER_RESERVE=1024
  export BC_VAL_MAX_SAMPLES=-1
  export BC_DISABLE_WANDB=1
  export WANDB_MODE=disabled
  export EXPERIMENT_NAME="$BC_EXPERIMENT"
  bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh
)
BC_RC=$?

echo "[1/2] BC-P finished with exit code $BC_RC; starting GAIA"
(
  export SLURM_JOB_NODELIST="$FULL_NODELIST"
  export EXPECTED_NUM_NODES="$NUM_ALLOC_NODES"
  export MODEL_PATH="$BASE_MODEL_PATH"
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
  export PROMPT_LENGTH=8192
  export RESPONSE_LENGTH=24576
  export CONTEXT_LENGTH=32768
  export TRAIN_BATCH_SIZE=32
  export PPO_MINI_BATCH_SIZE=16
  export ROLLOUT_N=1
  export MAX_TURN=100
  export MAX_SESSION=10
  export VAL_MAX_SESSION=10
  export TURN_MAX_NEW_TOKENS=2048
  export FINAL_ANSWER_RESERVE=1024
  export SESSION_TIMEOUT=3600
  export BC_SEARCH_TIMEOUT_SECONDS=600
  export BC_CTXGRAPH_PROTOCOL=controller
  export BC_CONTROLLER_ACTION_POLICY=balanced
  export BC_DISABLE_WANDB=1
  export WANDB_MODE=disabled
  bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
)
GAIA_RC=$?
set -e

echo "[2/2] GAIA finished with exit code $GAIA_RC"
echo "BC-P experiment: $BC_EXPERIMENT"
echo "GAIA experiment: $GAIA_EXPERIMENT"

if [ "$BC_RC" -ne 0 ] || [ "$GAIA_RC" -ne 0 ]; then
  echo "ERROR: one or more evaluations failed: BC-P=$BC_RC GAIA=$GAIA_RC" >&2
  exit 4
fi
