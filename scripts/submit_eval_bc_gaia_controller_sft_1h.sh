#!/bin/bash

# Submit the merged MiroVerse Qwen3-8B controller-SFT model for matched
# ContextGraph evaluation on BrowseComp-Plus and GAIA.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/09281/chc_1996}
MODEL_PATH=${SFT_EVAL_MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/998826_miroverse_qwen3_8b_lora32_4k_merged}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
BC_EVAL_TIME=${BC_EVAL_TIME:-01:00:00}
GAIA_EVAL_TIME=${GAIA_EVAL_TIME:-01:00:00}
DRY_RUN=${DRY_RUN:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac

if [ "$DRY_RUN" = "0" ]; then
  if [ ! -s "$MODEL_PATH/config.json" ]; then
    echo "ERROR: merged SFT model is missing config.json: $MODEL_PATH" >&2
    exit 2
  fi
  if ! find "$MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit | grep -q .; then
    echo "ERROR: merged SFT model has no non-empty Hugging Face weight files: $MODEL_PATH" >&2
    exit 2
  fi
fi

cd "$PROJECT_ROOT"
mkdir -p logs

LAST_JOB_ID=
submit_job() {
  local benchmark=$1
  shift
  local -a command=("$@")
  local output
  local job_id

  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY_RUN %s:' "$benchmark"
    printf ' %q' "${command[@]}"
    printf '\n'
    LAST_JOB_ID=DRY_RUN
    return
  fi

  output=$("${command[@]}")
  job_id=${output%%;*}
  if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse $benchmark job id from: $output" >&2
    exit 3
  fi
  LAST_JOB_ID=$job_id
  printf '%-8s job=%s\n' "$benchmark" "$job_id"
}

BC_EXPERIMENT="eval_bc_controller_miroverse_sft_4k_4n_64k_${STAMP}"
submit_job BC-P \
  sbatch --parsable \
  --job-name=eval-bc-controller-sft8b-64k \
  --nodes=4 \
  --time="$BC_EVAL_TIME" \
  --output='logs/eval-bc-controller-sft8b-64k.%j.out' \
  --error='logs/eval-bc-controller-sft8b-64k.%j.err' \
  --export="ALL,MODEL_PATH=$MODEL_PATH,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,BC_METHOD=contextgraph,BC_CTXGRAPH_PROTOCOL=controller,BC_CONTROLLER_ACTION_POLICY=structural,BC_EXPERIMENT_MODEL_TAG=miroverse_controller_sft_4k,BC_CONTEXT_LENGTH=65536,BC_PROMPT_LENGTH=8192,BC_RESPONSE_LENGTH=57344,BC_YARN_FACTOR=2.0,BC_YARN_ORIGINAL_LENGTH=32768,BC_FINAL_ANSWER_RESERVE=1024,BC_VAL_MAX_SAMPLES=-1,BC_DISABLE_WANDB=1,WANDB_MODE=disabled,EXPERIMENT_NAME=$BC_EXPERIMENT" \
  scripts/eval_bc_baseline_8b_4node_zeroshot.sh
BC_JOB_ID=$LAST_JOB_ID

GAIA_EXPERIMENT="eval_gaia_controller_miroverse_sft_4k_5n_32k_${STAMP}"
GAIA_OUTPUT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$GAIA_EXPERIMENT"
submit_job GAIA \
  sbatch --parsable \
  --job-name=eval-gaia-controller-sft8b-32k \
  --nodes=5 \
  --time="$GAIA_EVAL_TIME" \
  --output='logs/eval-gaia-controller-sft8b-32k.%j.out' \
  --error='logs/eval-gaia-controller-sft8b-32k.%j.err' \
  --export="ALL,MODEL_PATH=$MODEL_PATH,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,EXPECTED_NUM_NODES=5,EXPERIMENT_NAME=$GAIA_EXPERIMENT,CHECKPOINT_ROOT=$GAIA_OUTPUT_ROOT,TRAIN_DATA_FILE=data/gaia_validation_graph.parquet,VAL_DATA_FILE=data/gaia_validation_graph.parquet,TRAIN_MAX_SAMPLES=-1,VAL_MAX_SAMPLES=-1,TRAINER_VAL_ONLY=True,VAL_BEFORE_TRAIN=True,TOTAL_TRAINING_STEPS=1,TEST_FREQ=999,SAVE_FREQ=-1,PROMPT_LENGTH=8192,RESPONSE_LENGTH=24576,CONTEXT_LENGTH=32768,TRAIN_BATCH_SIZE=32,PPO_MINI_BATCH_SIZE=16,ROLLOUT_N=1,MAX_TURN=100,MAX_SESSION=10,VAL_MAX_SESSION=10,TURN_MAX_NEW_TOKENS=2048,FINAL_ANSWER_RESERVE=1024,SESSION_TIMEOUT=3600,BC_SEARCH_TIMEOUT_SECONDS=600,BC_CTXGRAPH_PROTOCOL=controller,BC_CONTROLLER_ACTION_POLICY=balanced,BC_DISABLE_WANDB=1,WANDB_MODE=disabled" \
  scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
GAIA_JOB_ID=$LAST_JOB_ID

if [ "$DRY_RUN" = "0" ]; then
  echo "Submitted BC-P and GAIA controller-SFT evaluations (one-hour walltime each)."
  squeue -j "$BC_JOB_ID,$GAIA_JOB_ID" -o '%.18i %.9P %.38j %.2t %.10M %.10L %.6D %R'
fi
