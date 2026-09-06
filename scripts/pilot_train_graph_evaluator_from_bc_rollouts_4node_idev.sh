#!/bin/bash
#SBATCH -J bc-graph-eval-pilot
#SBATCH -o logs/bc-graph-eval-pilot.%j.out
#SBATCH -e logs/bc-graph-eval-pilot.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 02:00:00
#SBATCH -A AST24021

# Train a small, independent GraphRPO evaluator from the completed BrowseComp
# rollout pilot. This is the bootstrap stage for alternating actor/evaluator
# training; it does not update the actor or touch bc_test.parquet.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
GRAPH_EVALUATOR_BASE_MODEL=${GRAPH_EVALUATOR_BASE_MODEL:-Qwen/Qwen3-0.6B}
GRAPH_EVALUATOR_MAX_QUESTIONS=${GRAPH_EVALUATOR_MAX_QUESTIONS:-0}
GRAPH_EVALUATOR_MAX_LENGTH=${GRAPH_EVALUATOR_MAX_LENGTH:-2048}
GRAPH_EVALUATOR_EPOCHS=${GRAPH_EVALUATOR_EPOCHS:-1}
GRAPH_EVALUATOR_LEARNING_RATE=${GRAPH_EVALUATOR_LEARNING_RATE:-2e-5}
GRAPH_EVALUATOR_BATCH_SIZE=${GRAPH_EVALUATOR_BATCH_SIZE:-1}
GRAPH_EVALUATOR_GRAD_ACCUM=${GRAPH_EVALUATOR_GRAD_ACCUM:-8}
GRAPH_EVALUATOR_VALIDATION_FRACTION=${GRAPH_EVALUATOR_VALIDATION_FRACTION:-0.2}
GRAPH_EVALUATOR_MIN_TRAIN_MIXED_QUESTIONS=${GRAPH_EVALUATOR_MIN_TRAIN_MIXED_QUESTIONS:-10}
GRAPH_EVALUATOR_MIN_VALIDATION_MIXED_QUESTIONS=${GRAPH_EVALUATOR_MIN_VALIDATION_MIXED_QUESTIONS:-5}
GRAPH_EVALUATOR_MIN_AUROC=${GRAPH_EVALUATOR_MIN_AUROC:-0.55}
GRAPH_EVALUATOR_MIN_WITHIN_QUESTION_AUROC=${GRAPH_EVALUATOR_MIN_WITHIN_QUESTION_AUROC:-0.55}
GRAPH_EVALUATOR_MAX_BRIER_RATIO=${GRAPH_EVALUATOR_MAX_BRIER_RATIO:-1.0}
GRAPH_EVALUATOR_SEED=${GRAPH_EVALUATOR_SEED:-42}
GRAPH_EVALUATOR_PROBE_PORT=${GRAPH_EVALUATOR_PROBE_PORT:-19002}
RUN_TS=$(date +%Y%m%d_%H%M%S)

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}

if [ -n "${WORK:-}" ] && [ -f "$WORK/.wandb_env" ]; then
  source "$WORK/.wandb_env"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is required; export it or place it in \$WORK/.wandb_env"
  exit 1
fi
export WANDB_MODE=online

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run inside the existing four-node idev allocation or submit with sbatch"
  exit 1
fi
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#NODELIST[@]}" -ne 4 ]; then
  echo "ERROR: expected exactly four allocated nodes; got ${#NODELIST[@]}"
  exit 1
fi
EVALUATOR_NODE=${NODELIST[0]}

if [ -z "${BOOTSTRAP_ROLLOUT_DIR:-}" ]; then
  BOOTSTRAP_ROLLOUT_DIR=$(find "$SCRATCH_ROOT/context-graph-rollouts" -mindepth 1 -maxdepth 1 -type d -name 'train_ctxgraph_bc_8b_graphrpo_ref_sft_4n_bs3_n8_20step_pilot_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-) || true
fi
if [ -z "${BOOTSTRAP_ROLLOUT_DIR:-}" ] || [ ! -d "$BOOTSTRAP_ROLLOUT_DIR" ]; then
  echo "ERROR: no completed BrowseComp rollout pilot was found; set BOOTSTRAP_ROLLOUT_DIR"
  exit 1
fi
mapfile -t ROLLOUT_FILES < <(find "$BOOTSTRAP_ROLLOUT_DIR" -maxdepth 1 -type f -name '*.jsonl' -size +0c -print | sort)
if [ "${#ROLLOUT_FILES[@]}" -lt 2 ]; then
  echo "ERROR: evaluator bootstrap needs multiple rollout JSONL files; found ${#ROLLOUT_FILES[@]}"
  exit 1
fi

RUN_NAME=${RUN_NAME:-bc_graph_evaluator_bootstrap_$RUN_TS}
DATA_DIR=${GRAPH_EVALUATOR_DATA_ROOT:-$SCRATCH_ROOT/context-graph-evaluator-data/$RUN_NAME}
MODEL_DIR=${GRAPH_EVALUATOR_MODEL_ROOT:-$SCRATCH_ROOT/context-graph-evaluators/$RUN_NAME}
TRAIN_LOG="$MODEL_DIR/train.log"
SERVER_LOG="$MODEL_DIR/server_probe.log"
mkdir -p "$PROJECT_ROOT/logs" "$DATA_DIR" "$MODEL_DIR"

echo "=============================================================="
echo "  BROWSECOMP GRAPH EVALUATOR BOOTSTRAP PILOT"
echo "  Source rollouts: $BOOTSTRAP_ROLLOUT_DIR"
echo "  JSONL files:     ${#ROLLOUT_FILES[@]}"
echo "  Training node:   $EVALUATOR_NODE"
echo "  Base model:      $GRAPH_EVALUATOR_BASE_MODEL"
echo "  Data output:     $DATA_DIR"
echo "  Model output:    $MODEL_DIR"
echo "  W&B run:         $RUN_NAME"
echo "=============================================================="

python scripts/audit_bc_judge_results.py "$BOOTSTRAP_ROLLOUT_DIR" --fail-on-integrity-error
python scripts/prepare_graph_evaluator_data.py "${ROLLOUT_FILES[@]}" --output-dir "$DATA_DIR" --validation-fraction "$GRAPH_EVALUATOR_VALIDATION_FRACTION" --seed "$GRAPH_EVALUATOR_SEED" --auto-seed-attempts 10000 --require-both-classes --min-train-mixed-questions "$GRAPH_EVALUATOR_MIN_TRAIN_MIXED_QUESTIONS" --min-validation-mixed-questions "$GRAPH_EVALUATOR_MIN_VALIDATION_MIXED_QUESTIONS" --max-questions "$GRAPH_EVALUATOR_MAX_QUESTIONS"

srun --overlap --nodes=1 --ntasks=1 -w "$EVALUATOR_NODE" --chdir="$PROJECT_ROOT" python scripts/train_graph_evaluator.py --train-file "$DATA_DIR/graph_evaluator_train.parquet" --validation-file "$DATA_DIR/graph_evaluator_validation.parquet" --model "$GRAPH_EVALUATOR_BASE_MODEL" --output-dir "$MODEL_DIR" --max-length "$GRAPH_EVALUATOR_MAX_LENGTH" --epochs "$GRAPH_EVALUATOR_EPOCHS" --learning-rate "$GRAPH_EVALUATOR_LEARNING_RATE" --batch-size "$GRAPH_EVALUATOR_BATCH_SIZE" --gradient-accumulation-steps "$GRAPH_EVALUATOR_GRAD_ACCUM" --seed "$GRAPH_EVALUATOR_SEED" --report-to wandb --wandb-project context-graph-evaluator --run-name "$RUN_NAME" 2>&1 | tee "$TRAIN_LOG"

if [ ! -s "$MODEL_DIR/config.json" ] || [ ! -s "$MODEL_DIR/graph_rpo_calibration.json" ] || [ ! -s "$MODEL_DIR/graph_rpo_evaluation.json" ] || [ ! -s "$MODEL_DIR/tokenizer_config.json" ] || ! find -L "$MODEL_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  echo "ERROR: evaluator checkpoint is incomplete: $MODEL_DIR"
  exit 1
fi

QUALITY_PASSED=1
python scripts/check_graph_evaluator_quality.py "$MODEL_DIR/graph_rpo_evaluation.json" --min-auroc "$GRAPH_EVALUATOR_MIN_AUROC" --max-brier-ratio "$GRAPH_EVALUATOR_MAX_BRIER_RATIO" --min-mixed-questions "$GRAPH_EVALUATOR_MIN_VALIDATION_MIXED_QUESTIONS" --min-within-question-auroc "$GRAPH_EVALUATOR_MIN_WITHIN_QUESTION_AUROC" || QUALITY_PASSED=0

srun --overlap --nodes=1 --ntasks=1 -w "$EVALUATOR_NODE" --chdir="$PROJECT_ROOT" python -u scripts/serve_graph_evaluator.py --model "$MODEL_DIR" --host 0.0.0.0 --port "$GRAPH_EVALUATOR_PROBE_PORT" --device cuda --max-length "$GRAPH_EVALUATOR_MAX_LENGTH" --local-files-only >"$SERVER_LOG" 2>&1 &
SERVER_STEP_PID=$!
cleanup_server() {
  kill "$SERVER_STEP_PID" 2>/dev/null || true
  wait "$SERVER_STEP_PID" 2>/dev/null || true
}
trap cleanup_server EXIT
EVALUATOR_NODE_IP=$(getent hosts "$EVALUATOR_NODE" | awk '{print $1}')
READY=0
for _ in $(seq 1 180); do
  if curl --noproxy '*' -fsS "http://${EVALUATOR_NODE_IP}:${GRAPH_EVALUATOR_PROBE_PORT}/health" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_STEP_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ "$READY" != "1" ]; then
  echo "ERROR: trained evaluator failed its HTTP health probe"
  tail -80 "$SERVER_LOG" || true
  exit 1
fi
curl --noproxy '*' -fsS -H 'Content-Type: application/json' -d '{"schema_version":"contextgraph.graph_evaluator.v1","items":[{"question":"BrowseComp evaluator pilot probe","graph_view":"[n1] root question"}]}' "http://${EVALUATOR_NODE_IP}:${GRAPH_EVALUATOR_PROBE_PORT}/score"
echo

echo "=============================================================="
echo "  EVALUATOR TRAINING COMPLETED"
echo "  Model:       $MODEL_DIR"
echo "  Data:        $DATA_DIR"
echo "  Evaluation:  $MODEL_DIR/graph_rpo_evaluation.json"
echo "  W&B project: context-graph-evaluator"
echo "  Quality gate:$QUALITY_PASSED"
echo "=============================================================="
if [ "$QUALITY_PASSED" != "1" ]; then
  echo "ERROR: evaluator trained successfully but did not pass the policy-use quality gate"
  exit 2
fi
