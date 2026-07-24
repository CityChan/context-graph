#!/bin/bash
set -euo pipefail

# Submit matched 8B BrowseComp training runs, then evaluate each resulting
# checkpoint on the corresponding text-only GAIA validation parquet.

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts}
TRAIN_TIME=${TRAIN_TIME:-24:00:00}
GAIA_EVAL_TIME=${GAIA_EVAL_TIME:-06:00:00}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

cd "$PROJECT_ROOT"

submit_method() {
  local method=$1
  local train_script=$2
  local gaia_data=$3
  local train_job_name=$4
  local eval_job_name=$5
  local train_exp="train_${method}_bc_8b_4n_gaia_transfer_${STAMP}"
  local train_root="$CHECKPOINT_BASE/$train_exp"
  local eval_exp="eval_gaia_${method}_from_${STAMP}"
  local train_out train_id eval_out eval_id

  train_out=$(sbatch \
    --job-name="$train_job_name" \
    --time="$TRAIN_TIME" \
    --export="ALL,EXPERIMENT_NAME=$train_exp,CHECKPOINT_ROOT=$train_root" \
    "$train_script")
  train_id=$(echo "$train_out" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$train_id" ]; then
    echo "ERROR: could not parse training job id from: $train_out" >&2
    exit 3
  fi

  eval_out=$(sbatch \
    --dependency="afterok:$train_id" \
    --job-name="$eval_job_name" \
    --time="$GAIA_EVAL_TIME" \
    --output="logs/${eval_job_name}.%j.out" \
    --error="logs/${eval_job_name}.%j.err" \
    --export="ALL,EXPERIMENT_NAME=$eval_exp,TRAIN_DATA_FILE=$gaia_data,VAL_DATA_FILE=$gaia_data,TRAINER_VAL_ONLY=True,TOTAL_TRAINING_STEPS=1,TEST_FREQ=999,SAVE_FREQ=-1,RESUME_CHECKPOINT_ROOT=$train_root" \
    "$train_script")
  eval_id=$(echo "$eval_out" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$eval_id" ]; then
    echo "ERROR: could not parse GAIA eval job id from: $eval_out" >&2
    exit 4
  fi

  echo "$method train_job=$train_id eval_job=$eval_id checkpoint=$train_root"
}

submit_method \
  baseline \
  scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh \
  data/gaia_validation.parquet \
  train-gaia-base-8b \
  eval-gaia-base-8b

submit_method \
  foldagent \
  scripts/train_bc_foldagent_8b_4node_24h_v3_32k.sh \
  data/gaia_validation_branch.parquet \
  train-gaia-fold-8b \
  eval-gaia-fold-8b

submit_method \
  ctxgraph \
  scripts/train_bc_ctxgraph_8b_4node_24h_v3_32k.sh \
  data/gaia_validation_graph.parquet \
  train-gaia-cg-8b \
  eval-gaia-cg-8b

squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.32j %.2t %.10M %.10L %.6D %R"
