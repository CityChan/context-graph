#!/bin/bash
#SBATCH -J eval-bc-gaia-sft8b
#SBATCH -o logs/eval-bc-gaia-controller-sft8b.%j.out
#SBATCH -e logs/eval-bc-gaia-controller-sft8b.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 01:00:00
#SBATCH -A AST24021

# Submit one five-node, one-hour job that evaluates only the merged MiroVerse
# controller-SFT model. BC-P runs first on four nodes; GAIA then uses all five.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/09281/chc_1996}
SFT_MODEL_PATH=${SFT_EVAL_MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/998826_miroverse_qwen3_8b_lora32_4k_merged}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
EVAL_TIME=${EVAL_TIME:-01:00:00}
DRY_RUN=${DRY_RUN:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SCRIPT_PATH=$(readlink -f "$0")

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

check_sft_model() {
  if [ ! -s "$SFT_MODEL_PATH/config.json" ]; then
    echo "ERROR: merged SFT model is missing config.json: $SFT_MODEL_PATH" >&2
    exit 2
  fi
  if ! find "$SFT_MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit | grep -q .; then
    echo "ERROR: merged SFT model has no non-empty Hugging Face weight files: $SFT_MODEL_PATH" >&2
    exit 2
  fi
}

cd "$PROJECT_ROOT"
mkdir -p logs

if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "$DRY_RUN" = "0" ]; then
    check_sft_model
  fi

  submit_command=(
    sbatch --parsable
    --nodes=5
    --time="$EVAL_TIME"
    --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,SCRATCH_ROOT=$SCRATCH_ROOT,SFT_EVAL_MODEL_PATH=$SFT_MODEL_PATH,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,STAMP=$STAMP,DRY_RUN=0"
    "$SCRIPT_PATH"
  )

  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY_RUN:'
    printf ' %q' "${submit_command[@]}"
    printf '\n'
    exit 0
  fi

  submission=$("${submit_command[@]}")
  job_id=${submission%%;*}
  if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse job id from: $submission" >&2
    exit 3
  fi
  echo "Submitted BC-P + GAIA SFT evaluation as one job: $job_id"
  squeue -j "$job_id" -o '%.18i %.9P %.38j %.2t %.10M %.10L %.6D %R' || true
  exit 0
fi

check_sft_model

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#ALLOC_NODES[@]}" -ne 5 ]; then
  echo "ERROR: expected five allocated nodes, got ${#ALLOC_NODES[@]}" >&2
  exit 4
fi

FULL_NODELIST=$SLURM_JOB_NODELIST
BC_NODELIST=$(printf '%s\n' "${ALLOC_NODES[@]:0:4}" | paste -sd, -)
BC_EXPERIMENT="eval_bc_controller_miroverse_sft_4k_4n_64k_${STAMP}"
GAIA_EXPERIMENT="eval_gaia_controller_miroverse_sft_4k_5n_32k_${STAMP}"
GAIA_OUTPUT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$GAIA_EXPERIMENT"

echo "=============================================================="
echo "  Sequential BC-P + GAIA controller-SFT evaluation"
echo "  Job: $SLURM_JOB_ID"
echo "  Model: $SFT_MODEL_PATH"
echo "  BC-P nodes: $BC_NODELIST"
echo "  GAIA nodes: $FULL_NODELIST"
echo "=============================================================="

echo "[1/2] Starting BC-P SFT evaluation"
set +e
(
  export SLURM_JOB_NODELIST="$BC_NODELIST"
  export EXPECTED_NUM_NODES=4
  export MODEL_PATH="$SFT_MODEL_PATH"
  export HF_HOME HF_HUB_CACHE
  export BC_METHOD=contextgraph
  export BC_CTXGRAPH_PROTOCOL=controller
  export BC_CONTROLLER_ACTION_POLICY=structural
  export BC_EXPERIMENT_MODEL_TAG=miroverse_controller_sft_4k
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
  export EXPECTED_NUM_NODES=5
  export MODEL_PATH="$SFT_MODEL_PATH"
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
  exit 5
fi
