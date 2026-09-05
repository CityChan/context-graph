#!/bin/bash
#SBATCH -J graphrpo-ref-20
#SBATCH -o logs/graphrpo-ref-20.%j.out
#SBATCH -e logs/graphrpo-ref-20.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# Twenty-step frozen-reference GraphRPO pilot from the completed MiroVerse SFT
# checkpoint. This is a stability/coverage pilot, not a formal benchmark run.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
RUN_TS=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"

export SFT_FSDP_CHECKPOINT=${SFT_FSDP_CHECKPOINT:-$SCRATCH_ROOT/contextgraph_sft_checkpoints/miroverse_full_policy_qwen3_8b_bs16_1ep_v1/global_step_174}
export MODEL_PATH=${MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/miroverse_full_policy_qwen3_8b_bs16_1ep_v1_step174_hf}
export EXPECTED_NUM_NODES=4
export RUN_TAG=${RUN_TAG:-graphrpo_ref_sft_4n_bs3_n8_20step_pilot}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-train_ctxgraph_bc_8b_${RUN_TAG}_${RUN_TS}}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH_ROOT/context-graph-ckpts/$EXPERIMENT_NAME}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$SCRATCH_ROOT/context-graph-rollouts/$EXPERIMENT_NAME}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-3}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-60}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-3}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=0
export SAVE_FREQ=${SAVE_FREQ:-20}
export SAVE_ROLLOUT_DATA=1
export BC_DISABLE_WANDB=0
export BC_REQUIRE_WANDB=1
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano}
export GRAPH_RPO_CREDIT_BACKEND=reference_answer_likelihood
export MAX_SESSION=${MAX_SESSION:-3}
export VAL_MAX_SESSION=${VAL_MAX_SESSION:-3}
export MAX_TURN=${MAX_TURN:-40}

if [ "$TOTAL_TRAINING_STEPS" -ne 20 ]; then
  echo "WARNING: this pilot is designed for 20 steps; requested $TOTAL_TRAINING_STEPS"
fi

mkdir -p "$PROJECT_ROOT/logs"
PILOT_LOG="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}.log"

echo "=============================================================="
echo "  FROZEN-REFERENCE GRAPHRPO 20-STEP PILOT"
echo "  Experiment:   $EXPERIMENT_NAME"
echo "  SFT model:    $MODEL_PATH"
echo "  Steps:        $TOTAL_TRAINING_STEPS"
echo "  Questions:    $TRAIN_MAX_SAMPLES"
echo "  Rollout n:    $ROLLOUT_N"
echo "  Checkpoints:  $CHECKPOINT_ROOT"
echo "  Rollout data: $ROLLOUT_DATA_DIR"
echo "  Main log:     $PILOT_LOG"
echo "=============================================================="

set +e
bash scripts/smoke_train_bc_ctxgraph_8b_graphrpo_5node_idev.sh 2>&1 | tee "$PILOT_LOG"
PILOT_RC=${PIPESTATUS[0]}
set -e

echo "PILOT_RC=$PILOT_RC"
if [ "$PILOT_RC" -ne 0 ]; then
  echo "ERROR: 20-step pilot failed; inspect $PILOT_LOG"
  exit "$PILOT_RC"
fi

mapfile -t ROLLOUT_FILES < <(find "$ROLLOUT_DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' -print | sort)
if [ "${#ROLLOUT_FILES[@]}" -eq 0 ]; then
  echo "ERROR: pilot exited successfully but wrote no rollout JSONL under $ROLLOUT_DATA_DIR"
  exit 1
fi

echo "Rollout JSONL files:"
wc -l "${ROLLOUT_FILES[@]}"
python scripts/audit_bc_judge_results.py "${ROLLOUT_FILES[@]}" --fail-on-integrity-error

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

echo "Key GraphRPO evidence from the final 20 metric records:"
grep -E 'step:[0-9]+|reference_creditable_edits|reference_scored_states|reference_delta_abs_sum|actor/skipped_nonfinite_micro_batch|actor/pg_loss|actor/grad_norm|actor/kl_loss' "$PILOT_LOG" | tail -20 || true

echo "=============================================================="
echo "  GRAPHRPO 20-STEP PILOT + JUDGE AUDIT COMPLETED"
echo "  Checkpoint:   $CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS"
echo "  Rollout data: $ROLLOUT_DATA_DIR"
echo "  Main log:     $PILOT_LOG"
echo "=============================================================="
