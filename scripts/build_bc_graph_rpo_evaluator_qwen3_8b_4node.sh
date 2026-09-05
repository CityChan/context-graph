#!/bin/bash
#SBATCH -J build-bc-graphrpo-eval-8b
#SBATCH -o logs/build-bc-graphrpo-eval-8b.%j.out
#SBATCH -e logs/build-bc-graphrpo-eval-8b.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 08:00:00
#SBATCH -A AST24021

# Build the frozen, target-domain GraphRPO evaluator in one workflow:
#   1. collect stochastic graph trajectories from the original Qwen3-8B policy
#      on BrowseComp-Plus train questions without performing a policy update;
#   2. audit the fixed BrowseComp judge decisions;
#   3. make a question-disjoint evaluator split;
#   4. fine-tune the cross-domain evaluator pilot and calibrate on held-out
#      BrowseComp questions;
#   5. save an independent evaluator checkpoint and run an HTTP score probe.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
TARGET_POLICY_MODEL=${TARGET_POLICY_MODEL:-/work/09281/chc_1996/vista/cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
TARGET_POLICY_DATA_FILE=${TARGET_POLICY_DATA_FILE:-$PROJECT_ROOT/data/bc_train.parquet}
GRAPH_EVALUATOR_PRETRAINED_MODEL=${GRAPH_EVALUATOR_PRETRAINED_MODEL:-}
GRAPH_EVALUATOR_BC_MAX_QUESTIONS=${GRAPH_EVALUATOR_BC_MAX_QUESTIONS:--1}
GRAPH_EVALUATOR_BC_ROLLOUTS_PER_QUESTION=${GRAPH_EVALUATOR_BC_ROLLOUTS_PER_QUESTION:-4}
GRAPH_EVALUATOR_VALIDATION_FRACTION=${GRAPH_EVALUATOR_VALIDATION_FRACTION:-0.2}
GRAPH_EVALUATOR_SEED=${GRAPH_EVALUATOR_SEED:-42}
GRAPH_EVALUATOR_MAX_LENGTH=${GRAPH_EVALUATOR_MAX_LENGTH:-2048}
GRAPH_EVALUATOR_EPOCHS=${GRAPH_EVALUATOR_EPOCHS:-1}
GRAPH_EVALUATOR_LEARNING_RATE=${GRAPH_EVALUATOR_LEARNING_RATE:-5e-6}
GRAPH_EVALUATOR_BATCH_SIZE=${GRAPH_EVALUATOR_BATCH_SIZE:-1}
GRAPH_EVALUATOR_GRAD_ACCUM=${GRAPH_EVALUATOR_GRAD_ACCUM:-8}
GRAPH_EVALUATOR_PROBE_PORT=${GRAPH_EVALUATOR_PROBE_PORT:-19002}
REUSE_TARGET_ROLLOUTS=${REUSE_TARGET_ROLLOUTS:-0}
export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano}
RUN_TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-bc_target_policy_qwen3_8b_evaldata_$RUN_TS}
TARGET_ROLLOUT_DIR=${TARGET_ROLLOUT_DIR:-$SCRATCH_ROOT/context-graph-evaluator-rollouts/$EXPERIMENT_NAME}
DATA_DIR=${GRAPH_EVALUATOR_DATA_ROOT:-$SCRATCH_ROOT/context-graph-evaluator-data/bc_qwen3_8b_$RUN_TS}
MODEL_DIR=${GRAPH_EVALUATOR_MODEL_ROOT:-$SCRATCH_ROOT/context-graph-evaluators/bc_qwen3_8b_frozen_$RUN_TS}
COLLECTION_LOG="$TARGET_ROLLOUT_DIR/collection.log"
TRAIN_LOG="$MODEL_DIR/train.log"
SERVER_LOG="$MODEL_DIR/server_probe.log"

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this script inside a four- or five-node Vista allocation, or submit it with sbatch."
  exit 1
fi
mapfile -t BUILD_NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "$REUSE_TARGET_ROLLOUTS" = "1" ] && [ "${#BUILD_NODELIST[@]}" -lt 1 ]; then
  echo "ERROR: resuming evaluator preparation requires at least one allocated node."
  exit 1
fi
if [ "$REUSE_TARGET_ROLLOUTS" != "1" ] && { [ "${#BUILD_NODELIST[@]}" -lt 4 ] || [ "${#BUILD_NODELIST[@]}" -gt 5 ]; }; then
  echo "ERROR: expected four or five allocated nodes; got ${#BUILD_NODELIST[@]}."
  exit 1
fi
export EXPECTED_NUM_NODES=${#BUILD_NODELIST[@]}
if [ "$GRAPH_EVALUATOR_BC_ROLLOUTS_PER_QUESTION" -lt 2 ]; then
  echo "ERROR: collect at least two independently sampled episodes per BrowseComp question."
  exit 1
fi

if [ ! -s "$TARGET_POLICY_DATA_FILE" ]; then
  echo "ERROR: BrowseComp training data is missing: $TARGET_POLICY_DATA_FILE"
  exit 1
fi
case "$(realpath -m "$TARGET_POLICY_DATA_FILE")" in
  *bc_test.parquet)
    echo "ERROR: evaluator collection must not use the BrowseComp test split."
    exit 1
    ;;
esac
if [ ! -s "$TARGET_POLICY_MODEL/config.json" ] || ! find -L "$TARGET_POLICY_MODEL" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: original Qwen3-8B target policy checkpoint is incomplete: $TARGET_POLICY_MODEL"
  exit 1
fi

if [ -z "$GRAPH_EVALUATOR_PRETRAINED_MODEL" ]; then
  GRAPH_EVALUATOR_PRETRAINED_MODEL=$(find "$SCRATCH_ROOT/context-graph-evaluators" -mindepth 1 -maxdepth 1 -type d -name 'sft_raw_pilot_qwen3_0p6b_*' -exec test -s '{}/graph_rpo_calibration.json' \; -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-) || true
fi
if [ -z "$GRAPH_EVALUATOR_PRETRAINED_MODEL" ] || [ ! -s "$GRAPH_EVALUATOR_PRETRAINED_MODEL/config.json" ] || [ ! -s "$GRAPH_EVALUATOR_PRETRAINED_MODEL/graph_rpo_calibration.json" ]; then
  echo "ERROR: no complete cross-domain evaluator pilot was found."
  echo "Set GRAPH_EVALUATOR_PRETRAINED_MODEL to the completed pilot checkpoint."
  exit 1
fi

RAW_SFT_REAL=$(realpath -m "$SCRATCH_ROOT/contextgraph_sft")
for OUTPUT_PATH in "$TARGET_ROLLOUT_DIR" "$DATA_DIR" "$MODEL_DIR"; do
  OUTPUT_REAL=$(realpath -m "$OUTPUT_PATH")
  case "$OUTPUT_REAL" in
    "$RAW_SFT_REAL"|"$RAW_SFT_REAL"/*)
      echo "ERROR: evaluator artifacts must not be written inside SFT data: $OUTPUT_REAL"
      exit 1
      ;;
  esac
done
mkdir -p "$TARGET_ROLLOUT_DIR" "$DATA_DIR" "$MODEL_DIR"

echo "=============================================================="
echo "  BUILD BROWSECOMP TARGET-DOMAIN GRAPHRPO EVALUATOR"
echo "  Target policy:       $TARGET_POLICY_MODEL"
echo "  Collection split:    $TARGET_POLICY_DATA_FILE"
echo "  Question cap:        $GRAPH_EVALUATOR_BC_MAX_QUESTIONS (-1 means all train questions)"
echo "  Episodes/question:   $GRAPH_EVALUATOR_BC_ROLLOUTS_PER_QUESTION"
echo "  Sampling:            temperature=1.0, do_sample=True"
echo "  Policy updates:      disabled (trainer.val_only=True)"
echo "  Pretrained evaluator:$GRAPH_EVALUATOR_PRETRAINED_MODEL"
echo "  Raw target rollouts: $TARGET_ROLLOUT_DIR"
echo "  Derived data:        $DATA_DIR"
echo "  Frozen model:        $MODEL_DIR"
echo "=============================================================="

if [ "$REUSE_TARGET_ROLLOUTS" != "1" ]; then
  export MODEL_PATH="$TARGET_POLICY_MODEL"
  export TRAIN_DATA_FILE="$TARGET_POLICY_DATA_FILE"
  export VAL_DATA_FILE="$TARGET_POLICY_DATA_FILE"
  export TRAIN_MAX_SAMPLES=$((EXPECTED_NUM_NODES - 1))
  export VAL_MAX_SAMPLES="$GRAPH_EVALUATOR_BC_MAX_QUESTIONS"
  export TRAIN_BATCH_SIZE=$((EXPECTED_NUM_NODES - 1))
  export PPO_MINI_BATCH_SIZE=$((EXPECTED_NUM_NODES - 1))
  export ROLLOUT_N=2
  export ROLLOUT_TEMPERATURE=1.0
  export VAL_ROLLOUT_N="$GRAPH_EVALUATOR_BC_ROLLOUTS_PER_QUESTION"
  export VAL_DO_SAMPLE=True
  export VAL_TEMPERATURE=1.0
  export TRAINER_VAL_ONLY=True
  export VAL_BEFORE_TRAIN=True
  export TOTAL_TRAINING_STEPS=1
  export TEST_FREQ=0
  export SAVE_FREQ=-1
  export SAVE_ROLLOUT_DATA=0
  export VALIDATION_DATA_DIR="$TARGET_ROLLOUT_DIR"
  export CHECKPOINT_ROOT="$SCRATCH_ROOT/context-graph-evaluator-collection-checkpoints/$EXPERIMENT_NAME"
  export ADV_ESTIMATOR=foldgrpo
  export POLICY_LOSS_MODE=vanilla
  export BC_CTXGRAPH_PROTOCOL=controller
  export BC_CONTROLLER_ACTION_POLICY=structural
  export PROCESS_REWARD_SPEC='[scope]'
  export BC_DISABLE_WANDB=1
  export RUN_TAG=graph_evaluator_collection
  bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh 2>&1 | tee "$COLLECTION_LOG"
fi

mapfile -t TARGET_RESULTS < <(find "$TARGET_ROLLOUT_DIR" -type f -name '*.jsonl' -size +0c -print | sort)
if [ "${#TARGET_RESULTS[@]}" -eq 0 ]; then
  echo "ERROR: no target-policy rollout JSONL was produced under $TARGET_ROLLOUT_DIR"
  exit 1
fi

python scripts/audit_bc_judge_results.py "$TARGET_ROLLOUT_DIR" --fail-on-integrity-error
python scripts/prepare_graph_evaluator_data.py "${TARGET_RESULTS[@]}" --output-dir "$DATA_DIR" --validation-fraction "$GRAPH_EVALUATOR_VALIDATION_FRACTION" --seed "$GRAPH_EVALUATOR_SEED" --auto-seed-attempts 10000 --require-both-classes
python -c 'import json,sys; from pathlib import Path; out=Path(sys.argv[1]); payload={"schema_version":"contextgraph.graph_evaluator_collection.v1","target_policy_model":sys.argv[2],"browsecomp_data_file":sys.argv[3],"episodes_per_question":int(sys.argv[4]),"sampling":{"do_sample":True,"temperature":1.0},"policy_updated":False,"judge_model":sys.argv[5],"pretrained_evaluator_model":sys.argv[6],"rollout_dir":sys.argv[7]}; out.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")' "$DATA_DIR/collection_manifest.json" "$TARGET_POLICY_MODEL" "$TARGET_POLICY_DATA_FILE" "$GRAPH_EVALUATOR_BC_ROLLOUTS_PER_QUESTION" "$JUDGE_MODEL" "$GRAPH_EVALUATOR_PRETRAINED_MODEL" "$TARGET_ROLLOUT_DIR"

python scripts/train_graph_evaluator.py --train-file "$DATA_DIR/graph_evaluator_train.parquet" --validation-file "$DATA_DIR/graph_evaluator_validation.parquet" --model "$GRAPH_EVALUATOR_PRETRAINED_MODEL" --output-dir "$MODEL_DIR" --max-length "$GRAPH_EVALUATOR_MAX_LENGTH" --epochs "$GRAPH_EVALUATOR_EPOCHS" --learning-rate "$GRAPH_EVALUATOR_LEARNING_RATE" --batch-size "$GRAPH_EVALUATOR_BATCH_SIZE" --gradient-accumulation-steps "$GRAPH_EVALUATOR_GRAD_ACCUM" --seed "$GRAPH_EVALUATOR_SEED" 2>&1 | tee "$TRAIN_LOG"

if [ ! -s "$MODEL_DIR/config.json" ] || [ ! -s "$MODEL_DIR/graph_rpo_calibration.json" ] || [ ! -s "$MODEL_DIR/graph_rpo_evaluation.json" ] || [ ! -s "$MODEL_DIR/tokenizer_config.json" ] || ! find -L "$MODEL_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: target-domain evaluator checkpoint is incomplete: $MODEL_DIR"
  exit 1
fi

python -u scripts/serve_graph_evaluator.py --model "$MODEL_DIR" --host 127.0.0.1 --port "$GRAPH_EVALUATOR_PROBE_PORT" --device cuda --max-length "$GRAPH_EVALUATOR_MAX_LENGTH" --local-files-only >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup_server() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup_server EXIT

READY=0
for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${GRAPH_EVALUATOR_PROBE_PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  echo "ERROR: target-domain evaluator failed its HTTP health probe."
  tail -80 "$SERVER_LOG" || true
  exit 1
fi
curl --noproxy '*' -fsS -H 'Content-Type: application/json' -d '{"schema_version":"contextgraph.graph_evaluator.v1","items":[{"question":"BrowseComp target-domain health probe","graph_view":"[n1] root question"}]}' "http://127.0.0.1:${GRAPH_EVALUATOR_PROBE_PORT}/score"
echo

echo "=============================================================="
echo "  BROWSECOMP TARGET-DOMAIN EVALUATOR BUILD COMPLETED"
echo "  GRAPH_RPO_EVALUATOR_MODEL=$MODEL_DIR"
echo "  Rollouts:   $TARGET_ROLLOUT_DIR"
echo "  Data:       $DATA_DIR"
echo "  Calibration:$MODEL_DIR/graph_rpo_calibration.json"
echo "  Evaluation: $MODEL_DIR/graph_rpo_evaluation.json"
echo "  Next: inspect held-out metrics, then freeze this exact path for GraphRPO."
echo "=============================================================="
