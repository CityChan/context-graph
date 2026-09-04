#!/bin/bash
# One-command four-node GraphRPO smoke from the original Qwen3-8B snapshot.
# This uses the deterministic graph evaluator and is for mechanics only.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
MODEL_SNAPSHOT=/work/09281/chc_1996/vista/cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUN_TS=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"

export MODEL_PATH="$MODEL_SNAPSHOT"
export EXPECTED_NUM_NODES=4
export RUN_TAG=graphrpo_qwen3_8b_zeroshot_4n_bs3_n8_judge_audit
export EXPERIMENT_NAME="train_ctxgraph_bc_8b_${RUN_TAG}_${RUN_TS}"
export CHECKPOINT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$EXPERIMENT_NAME"
export ROLLOUT_DATA_DIR="$SCRATCH_ROOT/context-graph-rollouts/$EXPERIMENT_NAME"
export TRAIN_BATCH_SIZE=3
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=2
export TRAIN_MAX_SAMPLES=3
export VAL_MAX_SAMPLES=3
export TOTAL_TRAINING_STEPS=1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=0
export SAVE_FREQ=1
export SAVE_ROLLOUT_DATA=1
export JUDGE_MODEL=gpt-5-nano

if [ ! -s "$MODEL_PATH/config.json" ] || ! find -L "$MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: original Qwen3-8B snapshot is incomplete: $MODEL_PATH"
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
SMOKE_LOG="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}.log"

echo "=============================================================="
echo "  ONE-COMMAND GRAPHRPO JUDGE-AUDIT SMOKE"
echo "  Experiment:   $EXPERIMENT_NAME"
echo "  Model:        $MODEL_PATH"
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

echo "Key GraphRPO evidence:"
grep -E 'TRAIN RUN COMPLETED|TRAIN RUN FAILED|graph_rpo_valid_edits|graph_rpo_scored_states|graph_rpo_delta_abs_sum|actor/pg_loss|actor/grad_norm|actor/kl_loss' "$SMOKE_LOG" | tail -20 || true

echo "=============================================================="
echo "  SMOKE + JUDGE AUDIT COMPLETED"
echo "  Checkpoint:   $CHECKPOINT_ROOT/global_step_1"
echo "  Rollout data: $ROLLOUT_DATA_DIR"
echo "  Main log:     $SMOKE_LOG"
echo "=============================================================="
