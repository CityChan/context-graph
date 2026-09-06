#!/bin/bash
# Two-step formal evaluator-free GraphRPO smoke. The pre-update Qwen3-8B
# policy answers paired before/after graph-conditioned QA probes, and the
# ordinary BrowseComp task judge supplies the utility difference.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
MODEL_SNAPSHOT=/work/09281/chc_1996/vista/cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
RUN_TS=$(date +%Y%m%d_%H%M%S)

cd "$PROJECT_ROOT"

export MODEL_PATH="$MODEL_SNAPSHOT"
export EXPECTED_NUM_NODES=4
export RUN_TAG=graphrpo_counterfactual_zeroshot_4n_bs3_n8_2step
export EXPERIMENT_NAME="train_ctxgraph_bc_8b_${RUN_TAG}_${RUN_TS}"
export CHECKPOINT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$EXPERIMENT_NAME"
export ROLLOUT_DATA_DIR="$SCRATCH_ROOT/context-graph-rollouts/$EXPERIMENT_NAME"
export TRAIN_BATCH_SIZE=3
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=2
export TRAIN_MAX_SAMPLES=6
export VAL_MAX_SAMPLES=3
export TOTAL_TRAINING_STEPS=2
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=0
export SAVE_FREQ=2
export SAVE_ROLLOUT_DATA=1
export JUDGE_MODEL=gpt-5-nano
export GRAPH_RPO_CREDIT_BACKEND=old_policy_counterfactual_qa
export GRAPH_RPO_COUNTERFACTUAL_SAMPLES=${GRAPH_RPO_COUNTERFACTUAL_SAMPLES:-1}
export GRAPH_RPO_COUNTERFACTUAL_MAX_NEW_TOKENS=${GRAPH_RPO_COUNTERFACTUAL_MAX_NEW_TOKENS:-512}
export GRAPH_RPO_ALPHA=0.1
export GRAPH_RPO_DELTA_MAX=0.25
export BC_DISABLE_WANDB=0
export BC_REQUIRE_WANDB=1

if [ ! -s "$MODEL_PATH/config.json" ] || ! find -L "$MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: original Qwen3-8B snapshot is incomplete: $MODEL_PATH"
  exit 1
fi

mkdir -p "$PROJECT_ROOT/logs"
SMOKE_LOG="$PROJECT_ROOT/logs/${EXPERIMENT_NAME}.log"

echo "=============================================================="
echo "  PAIRED-COUNTERFACTUAL GRAPHRPO TWO-STEP SMOKE"
echo "  Experiment:   $EXPERIMENT_NAME"
echo "  Model:        $MODEL_PATH"
echo "  Probe samples:$GRAPH_RPO_COUNTERFACTUAL_SAMPLES per graph state"
echo "  Rollout data: $ROLLOUT_DATA_DIR"
echo "  Main log:     $SMOKE_LOG"
echo "=============================================================="

set +e
bash scripts/smoke_train_bc_ctxgraph_8b_graphrpo_5node_idev.sh 2>&1 | tee "$SMOKE_LOG"
SMOKE_RC=${PIPESTATUS[0]}
set -e

if [ "$SMOKE_RC" -ne 0 ]; then
  echo "ERROR: smoke failed with rc=$SMOKE_RC; inspect $SMOKE_LOG"
  exit "$SMOKE_RC"
fi

mapfile -t ROLLOUT_FILES < <(find "$ROLLOUT_DATA_DIR" -maxdepth 1 -type f -name '*.jsonl' -print | sort)
if [ "${#ROLLOUT_FILES[@]}" -ne 2 ]; then
  echo "ERROR: expected two rollout JSONL files, found ${#ROLLOUT_FILES[@]}"
  exit 1
fi
if [ ! -d "$CHECKPOINT_ROOT/global_step_2/actor" ]; then
  echo "ERROR: optimizer checkpoint is missing: $CHECKPOINT_ROOT/global_step_2/actor"
  exit 1
fi

python scripts/audit_bc_judge_results.py "${ROLLOUT_FILES[@]}" --fail-on-integrity-error

grep -Eq 'reward/graph_rpo_creditable_edits:[1-9]' "$SMOKE_LOG" || { echo "ERROR: no counterfactual graph edits received credit"; exit 1; }
grep -Eq 'reward/graph_rpo_counterfactual_scored_states:[1-9]' "$SMOKE_LOG" || { echo "ERROR: counterfactual graph states were not scored"; exit 1; }
grep -Eq 'reward/graph_rpo_counterfactual_probe_rollouts:[1-9]' "$SMOKE_LOG" || { echo "ERROR: no counterfactual QA probes were sampled"; exit 1; }
grep -q 'actor/pg_loss:' "$SMOKE_LOG" || { echo "ERROR: actor policy loss was not logged"; exit 1; }
grep -q 'actor/grad_norm:' "$SMOKE_LOG" || { echo "ERROR: actor optimizer step was not logged"; exit 1; }

echo "Key GraphRPO evidence:"
grep -E 'training/global_step:|graph_rpo_creditable_edits|graph_rpo_counterfactual_(scored_states|probe_rollouts|delta_abs_sum)|actor/pg_loss|actor/grad_norm|actor/kl_loss' "$SMOKE_LOG" | tail -20

echo "=============================================================="
echo "  PAIRED-COUNTERFACTUAL GRAPHRPO TWO-STEP SMOKE COMPLETED"
echo "  Checkpoint:   $CHECKPOINT_ROOT/global_step_2"
echo "  Rollout data: $ROLLOUT_DATA_DIR"
echo "  Main log:     $SMOKE_LOG"
echo "=============================================================="
