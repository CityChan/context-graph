#!/bin/bash
set -euo pipefail

# Submit a matched three-way GAIA text-only benchmark for Qwen3-8B.
#
# Default (immediately runnable): architecture-only zero-shot comparison.
#   bash scripts/submit_gaia_benchmark_8b_5node.sh
#
# Checkpoint-transfer comparison (all three exact checkpoints are required):
#   GAIA_EVAL_MODE=checkpoint \
#   GAIA_BASELINE_CHECKPOINT=/scratch/.../global_step_N \
#   GAIA_FOLDAGENT_CHECKPOINT=/scratch/.../global_step_N \
#   GAIA_CTXGRAPH_CHECKPOINT=/scratch/.../global_step_N \
#   bash scripts/submit_gaia_benchmark_8b_5node.sh

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts}
GAIA_EVAL_MODE=${GAIA_EVAL_MODE:-zeroshot}
GAIA_EVAL_TIME=${GAIA_EVAL_TIME:-03:00:00}
GAIA_NUM_NODES=${GAIA_NUM_NODES:-5}
GAIA_METHODS=${GAIA_METHODS:-baseline,foldagent,ctxgraph}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

PROMPT_LENGTH=${PROMPT_LENGTH:-8192}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-24576}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
MAX_TURN=${MAX_TURN:-100}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-2048}
SESSION_TIMEOUT=${SESSION_TIMEOUT:-3600}
FINAL_ANSWER_RESERVE=${FINAL_ANSWER_RESERVE:-1024}
BC_YARN_FACTOR=${BC_YARN_FACTOR:-2.0}
BC_YARN_ORIGINAL_LENGTH=${BC_YARN_ORIGINAL_LENGTH:-32768}

if [ $((PROMPT_LENGTH + RESPONSE_LENGTH)) -ne "$CONTEXT_LENGTH" ]; then
  echo "ERROR: PROMPT_LENGTH + RESPONSE_LENGTH must equal CONTEXT_LENGTH" >&2
  exit 2
fi
CONTEXT_TAG="$((CONTEXT_LENGTH / 1024))k"

cd "$PROJECT_ROOT"
mkdir -p logs

for data_file in \
  data/gaia_validation.parquet \
  data/gaia_validation_branch.parquet \
  data/gaia_validation_graph.parquet; do
  if [ ! -s "$data_file" ]; then
    echo "ERROR: missing or empty GAIA parquet: $PROJECT_ROOT/$data_file" >&2
    exit 2
  fi
done

case "$GAIA_EVAL_MODE" in
  zeroshot)
    BASELINE_CHECKPOINT=""
    FOLDAGENT_CHECKPOINT=""
    CTXGRAPH_CHECKPOINT=""
    ;;
  checkpoint)
    : "${GAIA_BASELINE_CHECKPOINT:?set GAIA_BASELINE_CHECKPOINT to an exact global_step_N directory}"
    : "${GAIA_FOLDAGENT_CHECKPOINT:?set GAIA_FOLDAGENT_CHECKPOINT to an exact global_step_N directory}"
    : "${GAIA_CTXGRAPH_CHECKPOINT:?set GAIA_CTXGRAPH_CHECKPOINT to an exact global_step_N directory}"
    BASELINE_CHECKPOINT=$GAIA_BASELINE_CHECKPOINT
    FOLDAGENT_CHECKPOINT=$GAIA_FOLDAGENT_CHECKPOINT
    CTXGRAPH_CHECKPOINT=$GAIA_CTXGRAPH_CHECKPOINT
    for checkpoint_path in "$BASELINE_CHECKPOINT" "$FOLDAGENT_CHECKPOINT" "$CTXGRAPH_CHECKPOINT"; do
      if [ ! -d "$checkpoint_path" ]; then
        echo "ERROR: checkpoint directory does not exist: $checkpoint_path" >&2
        exit 3
      fi
    done
    ;;
  *)
    echo "ERROR: GAIA_EVAL_MODE must be zeroshot or checkpoint, got: $GAIA_EVAL_MODE" >&2
    exit 4
    ;;
esac

submit_method() {
  local method=$1
  local batch_script=$2
  local gaia_data=$3
  local checkpoint_path=$4
  local job_name="gaia-${method}-8b-${CONTEXT_TAG}-${GAIA_EVAL_MODE}"
  local experiment_name="eval_gaia_${method}_qwen3_8b_${CONTEXT_TAG}_${GAIA_EVAL_MODE}_${STAMP}"
  local output_root="$CHECKPOINT_BASE/$experiment_name"
  local export_vars
  local submit_output
  local job_id

  export_vars="ALL,EXPECTED_NUM_NODES=$GAIA_NUM_NODES,EXPERIMENT_NAME=$experiment_name,CHECKPOINT_ROOT=$output_root,TRAIN_DATA_FILE=$gaia_data,VAL_DATA_FILE=$gaia_data,TRAINER_VAL_ONLY=True,VAL_BEFORE_TRAIN=True,TOTAL_TRAINING_STEPS=1,TEST_FREQ=999,SAVE_FREQ=-1,PROMPT_LENGTH=$PROMPT_LENGTH,RESPONSE_LENGTH=$RESPONSE_LENGTH,CONTEXT_LENGTH=$CONTEXT_LENGTH,BC_YARN_FACTOR=$BC_YARN_FACTOR,BC_YARN_ORIGINAL_LENGTH=$BC_YARN_ORIGINAL_LENGTH,TRAIN_BATCH_SIZE=32,PPO_MINI_BATCH_SIZE=16,ROLLOUT_N=1,MAX_TURN=$MAX_TURN,MAX_SESSION=10,VAL_MAX_SESSION=10,TURN_MAX_NEW_TOKENS=$TURN_MAX_NEW_TOKENS,FINAL_ANSWER_RESERVE=$FINAL_ANSWER_RESERVE,SESSION_TIMEOUT=$SESSION_TIMEOUT,BC_SEARCH_TIMEOUT_SECONDS=600"
  if [ -n "$checkpoint_path" ]; then
    export_vars="$export_vars,RESUME_CHECKPOINT_PATH=$checkpoint_path"
  fi

  submit_output=$(sbatch \
    --job-name="$job_name" \
    --nodes="$GAIA_NUM_NODES" \
    --time="$GAIA_EVAL_TIME" \
    --output="logs/${job_name}.%j.out" \
    --error="logs/${job_name}.%j.err" \
    --export="$export_vars" \
    "$batch_script")
  job_id=$(printf '%s\n' "$submit_output" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$job_id" ]; then
    echo "ERROR: could not parse job id from: $submit_output" >&2
    exit 5
  fi

  printf '%-10s job=%s data=%s checkpoint=%s\n' "$method" "$job_id" "$gaia_data" "${checkpoint_path:-Qwen/Qwen3-8B}"
}

method_enabled() {
  case ",$GAIA_METHODS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

echo "Submitting matched GAIA benchmark: methods=$GAIA_METHODS mode=$GAIA_EVAL_MODE nodes=$GAIA_NUM_NODES prompt=$PROMPT_LENGTH response=$RESPONSE_LENGTH context=$CONTEXT_LENGTH final_reserve=$FINAL_ANSWER_RESERVE"

if method_enabled baseline; then
  submit_method \
    baseline \
    scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh \
    data/gaia_validation.parquet \
    "$BASELINE_CHECKPOINT"
fi

if method_enabled foldagent; then
  submit_method \
    foldagent \
    scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh \
    data/gaia_validation_branch.parquet \
    "$FOLDAGENT_CHECKPOINT"
fi

if method_enabled ctxgraph; then
  submit_method \
    ctxgraph \
    scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh \
    data/gaia_validation_graph.parquet \
    "$CTXGRAPH_CHECKPOINT"
fi

squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.32j %.2t %.10M %.10L %.6D %R"
