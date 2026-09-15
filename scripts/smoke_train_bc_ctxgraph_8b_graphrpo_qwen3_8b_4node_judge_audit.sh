#!/bin/bash
#SBATCH -J graphrpo-scale-smoke
#SBATCH -o logs/graphrpo-scale-smoke.%j.out
#SBATCH -e logs/graphrpo-scale-smoke.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 02:00:00
#SBATCH -A AST24021

# One-command four-node GraphRPO smoke from the original Qwen3-8B snapshot.
# Configuration: 1 search + 3 trainer nodes, batch 6, rollout n=8, LoRA-32.
# This uses the frozen original Qwen3-8B as the answer-likelihood reference.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
MODEL_SNAPSHOT=/work/09281/chc_1996/vista/cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUN_TS=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"

export MODEL_PATH="$MODEL_SNAPSHOT"
export EXPECTED_NUM_NODES=4
export RUN_TAG=${RUN_TAG:-graphrpo_qwen3_8b_zeroshot_4n_bs6_n8_lora32_judge_audit}
export EXPERIMENT_NAME="train_ctxgraph_bc_8b_${RUN_TAG}_${RUN_TS}"
export CHECKPOINT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$EXPERIMENT_NAME"
export ROLLOUT_DATA_DIR="$SCRATCH_ROOT/context-graph-rollouts/$EXPERIMENT_NAME"
export TRAIN_BATCH_SIZE=6
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=3
export TRAIN_MAX_SAMPLES=6
export VAL_MAX_SAMPLES=6
export LORA_RANK=32
export LORA_ALPHA=32
export LORA_TARGET_MODULES=all-linear
export TRAIN_LR=1e-5
export PROMPT_LENGTH=8192
export RESPONSE_LENGTH=24576
export CONTEXT_LENGTH=32768
export DATALOADER_NUM_WORKERS=0
export TOTAL_TRAINING_STEPS=1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=0
export SAVE_FREQ=1
export SAVE_ROLLOUT_DATA=1
export JUDGE_MODEL=gpt-5-nano
export GRAPH_RPO_CREDIT_BACKEND=reference_answer_likelihood
export GRAPH_RPO_ALPHA=0.1
export GRAPH_RPO_BETA=1.0
export GRAPH_RPO_DELTA_SCALE=${GRAPH_RPO_DELTA_SCALE:-1.0}
export GRAPH_RPO_DELTA_MAX=${GRAPH_RPO_DELTA_MAX:-0.25}
export GRAPH_RPO_AUDIT_MAX_CLIP_RATE=${GRAPH_RPO_AUDIT_MAX_CLIP_RATE:-1.0}
export USE_KL_LOSS=True
export ACTOR_KL_LOSS_COEF=0.0005
export ALGORITHM_KL_COEF=0.005
export MAX_SESSION=3
export VAL_MAX_SESSION=3
export MAX_TURN=40
export BC_REQUIRE_WANDB=1
export TRAINER_RESUME_MODE=disable

# A smoke must start from the immutable base snapshot, even if the caller's
# shell still contains resume variables from a previous run.
unset RESUME_CHECKPOINT_PATH RESUME_CHECKPOINT_ROOT

if [ ! -s "$MODEL_PATH/config.json" ] || ! find -L "$MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: original Qwen3-8B snapshot is incomplete: $MODEL_PATH"
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
SMOKE_LOG="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}.log"
FINAL_CHECKPOINT="$CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS"
ADAPTER_DIR="$FINAL_CHECKPOINT/actor/lora_adapter"
GRAPH_AUDIT_OUTPUT="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}.graph_credit_audit.json"

echo "=============================================================="
echo "  ONE-COMMAND GRAPHRPO JUDGE-AUDIT SMOKE"
echo "  Experiment:   $EXPERIMENT_NAME"
echo "  Model:        $MODEL_PATH"
echo "  Optimization: GraphRPO + frozen-reference credit, LoRA-32, BS=6, n=8"
echo "  Credit scale: delta_scale=$GRAPH_RPO_DELTA_SCALE, delta_max=$GRAPH_RPO_DELTA_MAX"
echo "  Rollout data: $ROLLOUT_DATA_DIR"
echo "  Main log:     $SMOKE_LOG"
echo "=============================================================="

set +e
bash scripts/smoke_train_bc_ctxgraph_8b_graphrpo_5node_idev.sh 2>&1 | tee "$SMOKE_LOG"
SMOKE_RC=${PIPESTATUS[0]}
set -e

echo "SMOKE_RC=$SMOKE_RC"
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "ERROR: smoke failed; inspect $SMOKE_LOG"
  exit "$SMOKE_RC"
fi

mapfile -t ROLLOUT_FILES < <(find "$ROLLOUT_DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' -print | sort)
if [ "${#ROLLOUT_FILES[@]}" -eq 0 ]; then
  echo "ERROR: smoke exited successfully but wrote no rollout JSONL under $ROLLOUT_DATA_DIR"
  exit 1
fi

echo "Rollout JSONL files:"
wc -l "${ROLLOUT_FILES[@]}"
python scripts/audit_bc_judge_results.py "${ROLLOUT_FILES[@]}" --fail-on-integrity-error
python scripts/audit_counterfactual_graph_credit.py "${ROLLOUT_FILES[@]}" --backend reference_answer_likelihood --output "$GRAPH_AUDIT_OUTPUT" --fail-on-integrity-error --fail-on-semantic-noop --require-nonzero-delta --expected-delta-scale "$GRAPH_RPO_DELTA_SCALE" --max-clip-rate "$GRAPH_RPO_AUDIT_MAX_CLIP_RATE"

if [ ! -s "$ADAPTER_DIR/adapter_config.json" ] || [ ! -s "$ADAPTER_DIR/adapter_model.safetensors" ]; then
  echo "ERROR: LoRA checkpoint is incomplete under $ADAPTER_DIR"
  exit 1
fi
if ! grep -q 'Launching ContextGraph graphrpo' "$SMOKE_LOG"; then
  echo "ERROR: trainer log does not confirm the GraphRPO estimator"
  exit 1
fi
if ! grep -q 'GraphRPO credit: reference_answer_likelihood' "$SMOKE_LOG"; then
  echo "ERROR: trainer log does not confirm frozen-reference GraphRPO credit"
  exit 1
fi
if ! grep -Eq 'reference_creditable_edits:[1-9]' "$SMOKE_LOG"; then
  echo "ERROR: no creditable frozen-reference graph edit was produced"
  exit 1
fi
if ! grep -Eq 'reference_scored_states:[1-9]' "$SMOKE_LOG"; then
  echo "ERROR: no frozen-reference graph states were scored"
  exit 1
fi
if ! grep -Eq 'reference_delta_abs_sum:(0\.[0-9]*[1-9]|[1-9])' "$SMOKE_LOG"; then
  echo "ERROR: all frozen-reference graph-edit deltas were zero"
  exit 1
fi

echo "Key GraphRPO evidence:"
grep -E 'TRAIN RUN COMPLETED|TRAIN RUN FAILED|graph_rpo_valid_edits|reference_creditable_edits|reference_scored_states|reference_delta_abs_sum|actor/skipped_nonfinite_micro_batch|actor/pg_loss|actor/grad_norm|actor/kl_loss' "$SMOKE_LOG" | tail -20 || true

echo "=============================================================="
echo "  SMOKE + JUDGE AUDIT COMPLETED"
echo "  Checkpoint:   $FINAL_CHECKPOINT"
echo "  LoRA adapter: $ADAPTER_DIR"
echo "  Rollout data: $ROLLOUT_DATA_DIR"
echo "  Graph audit:  $GRAPH_AUDIT_OUTPUT"
echo "  Main log:     $SMOKE_LOG"
echo "=============================================================="
