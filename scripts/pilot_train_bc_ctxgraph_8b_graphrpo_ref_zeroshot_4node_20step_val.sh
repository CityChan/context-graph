#!/bin/bash
#SBATCH -J graphrpo-zs-val
#SBATCH -o logs/graphrpo-zs-val.%j.out
#SBATCH -e logs/graphrpo-zs-val.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# Twenty-step frozen-reference GraphRPO from the original Qwen3-8B model,
# with matched full held-out validation before training and at step 20.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/work/09281/chc_1996/vista/cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
RUN_TS=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"

export MODEL_PATH=$BASE_MODEL_PATH
export EXPECTED_NUM_NODES=4
export RUN_TAG=${RUN_TAG:-graphrpo_ref_zeroshot_4n_bs6_n8_lora32_20step_val}
export EXPERIMENT_NAME="train_ctxgraph_bc_8b_${RUN_TAG}_${RUN_TS}"
export CHECKPOINT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$EXPERIMENT_NAME"
export ROLLOUT_DATA_DIR="$SCRATCH_ROOT/context-graph-rollouts/$EXPERIMENT_NAME"
export VALIDATION_DATA_DIR="$SCRATCH_ROOT/context-graph-validation/$EXPERIMENT_NAME"
export DATA_SEED=${DATA_SEED:-42}
export TRAIN_BATCH_SIZE=6
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=3
export TRAIN_MAX_SAMPLES=120
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
export LORA_RANK=32
export LORA_ALPHA=32
export LORA_TARGET_MODULES=all-linear
export TRAIN_LR=1e-5
export PROMPT_LENGTH=8192
export RESPONSE_LENGTH=24576
export CONTEXT_LENGTH=32768
export DATALOADER_NUM_WORKERS=0
export USE_KL_LOSS=True
export ACTOR_KL_LOSS_COEF=0.0005
export ALGORITHM_KL_COEF=0.005
export VAL_ROLLOUT_N=1
export VAL_DO_SAMPLE=False
export VAL_TEMPERATURE=0.0
export TOTAL_TRAINING_STEPS=20
export VAL_BEFORE_TRAIN=True
export TEST_FREQ=20
export SAVE_FREQ=5
export SAVE_ROLLOUT_DATA=1
export BC_DISABLE_WANDB=0
export BC_REQUIRE_WANDB=1
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano}
export GRAPH_RPO_CREDIT_BACKEND=reference_answer_likelihood
export GRAPH_RPO_ALPHA=0.1
export GRAPH_RPO_BETA=1.0
export GRAPH_RPO_DELTA_MAX=0.25
export TRAINER_RESUME_MODE=disable
export MAX_SESSION=${MAX_SESSION:-3}
export VAL_MAX_SESSION=${VAL_MAX_SESSION:-3}
export MAX_TURN=${MAX_TURN:-40}
unset RESUME_CHECKPOINT_PATH RESUME_CHECKPOINT_ROOT

for output_dir in "$CHECKPOINT_ROOT" "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR"; do
  if [ -e "$output_dir" ]; then
    echo "ERROR: isolated GraphRPO output path already exists: $output_dir"
    exit 1
  fi
done

mkdir -p "$PROJECT_ROOT/logs"
PILOT_LOG="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}.log"

echo "=============================================================="
echo "  ZERO-SHOT FROZEN-REFERENCE GRAPHRPO + MATCHED VALIDATION"
echo "  Experiment:       $EXPERIMENT_NAME"
echo "  Base model:       $MODEL_PATH"
echo "  Steps:            $TOTAL_TRAINING_STEPS"
echo "  Training rows:    $TRAIN_MAX_SAMPLES"
echo "  Training:         BS=$TRAIN_BATCH_SIZE, n=$ROLLOUT_N, LoRA=$LORA_RANK, context=$CONTEXT_LENGTH"
echo "  Validation rows:  $VAL_MAX_SAMPLES (-1 means all)"
echo "  Validation:       step 0 and step 20, greedy n=1"
echo "  Data seed:        $DATA_SEED"
echo "  Checkpoints:      $CHECKPOINT_ROOT"
echo "  Rollout data:     $ROLLOUT_DATA_DIR"
echo "  Validation data:  $VALIDATION_DATA_DIR"
echo "  Main log:         $PILOT_LOG"
echo "=============================================================="

set +e
bash scripts/smoke_train_bc_ctxgraph_8b_graphrpo_5node_idev.sh 2>&1 | tee "$PILOT_LOG"
PILOT_RC=${PIPESTATUS[0]}
set -e

echo "PILOT_RC=$PILOT_RC"
if [ "$PILOT_RC" -ne 0 ]; then
  echo "ERROR: zero-shot GraphRPO comparison failed; inspect $PILOT_LOG"
  exit "$PILOT_RC"
fi
if grep -Eq 'Found checkpoint:|Load from checkpoint folder:|Resuming from ' "$PILOT_LOG"; then
  echo "ERROR: isolated zero-shot GraphRPO run unexpectedly resumed a checkpoint"
  exit 1
fi

for step in 0 20; do
  VAL_FILE="$VALIDATION_DATA_DIR/$step.jsonl"
  if [ ! -s "$VAL_FILE" ]; then
    echo "ERROR: missing validation generations for step $step: $VAL_FILE"
    exit 1
  fi
  echo "Validation judge audit for step $step:"
  python scripts/audit_bc_judge_results.py "$VAL_FILE" --fail-on-integrity-error
done

mapfile -t ROLLOUT_FILES < <(find "$ROLLOUT_DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' -print | sort)
if [ "${#ROLLOUT_FILES[@]}" -ne 20 ]; then
  echo "ERROR: expected 20 training rollout files, found ${#ROLLOUT_FILES[@]}"
  exit 1
fi
echo "Training rollout JSONL files:"
wc -l "${ROLLOUT_FILES[@]}"
python scripts/audit_bc_judge_results.py "${ROLLOUT_FILES[@]}" --fail-on-integrity-error

ADAPTER_DIR="$CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS/actor/lora_adapter"
if [ ! -s "$ADAPTER_DIR/adapter_config.json" ] || [ ! -s "$ADAPTER_DIR/adapter_model.safetensors" ]; then
  echo "ERROR: final LoRA adapter checkpoint is incomplete under $ADAPTER_DIR"
  exit 1
fi

if ! grep -q 'Initial validation metrics:' "$PILOT_LOG"; then
  echo "ERROR: initial validation metrics were not logged"
  exit 1
fi
if ! grep -q 'Final validation metrics:' "$PILOT_LOG"; then
  echo "ERROR: final validation metrics were not logged"
  exit 1
fi
if ! grep -Eq 'reference_creditable_edits:[1-9]' "$PILOT_LOG"; then
  echo "ERROR: no step produced a creditable frozen-reference graph edit"
  exit 1
fi
if ! grep -Eq 'reference_scored_states:[1-9]' "$PILOT_LOG"; then
  echo "ERROR: no frozen-reference graph states were scored"
  exit 1
fi
if ! grep -Eq 'reference_delta_abs_sum:(0\.[0-9]*[1-9]|[1-9])' "$PILOT_LOG"; then
  echo "ERROR: all frozen-reference graph-edit deltas were zero"
  exit 1
fi

echo "Matched validation metrics:"
grep -E 'Initial validation metrics:|Final validation metrics:' "$PILOT_LOG"
echo "Key GraphRPO evidence from the final 20 metric records:"
grep -E 'step:[0-9]+|reference_creditable_edits|reference_scored_states|reference_delta_abs_sum|actor/skipped_nonfinite_micro_batch|actor/pg_loss|actor/grad_norm|actor/kl_loss' "$PILOT_LOG" | tail -20 || true

echo "=============================================================="
echo "  ZERO-SHOT GRAPHRPO 20-STEP + MATCHED VALIDATION COMPLETED"
echo "  Checkpoint:       $CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS"
echo "  Rollout data:     $ROLLOUT_DATA_DIR"
echo "  Validation data:  $VALIDATION_DATA_DIR"
echo "  W&B experiment:   $EXPERIMENT_NAME"
echo "  Main log:         $PILOT_LOG"
echo "=============================================================="
