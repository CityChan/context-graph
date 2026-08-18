#!/bin/bash
#SBATCH -J d3gym-30b
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/d3gym-30b.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/d3gym-30b.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH -A AST24021

# D3-Gym entrypoint for Qwen3-30B-A3B-Instruct-2507.  It reuses the tested
# multi-node Ray/FSDP launcher and swaps only the benchmark data and runtime.
#
# Examples:
#   D3GYM_MODE=smoke D3GYM_METHOD=react sbatch scripts/run_d3gym_30b_instruct_8node.sh
#   D3GYM_MODE=train D3GYM_METHOD=fold sbatch scripts/run_d3gym_30b_instruct_8node.sh
#   D3GYM_MODE=train D3GYM_METHOD=ctxgraph sbatch scripts/run_d3gym_30b_instruct_8node.sh

set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"

D3GYM_MODE=${D3GYM_MODE:-smoke}
D3GYM_METHOD=${D3GYM_METHOD:-react}
export D3GYM_RUNTIME=${D3GYM_RUNTIME:-apptainer}
export D3GYM_IMAGE_DIR=${D3GYM_IMAGE_DIR:-${SCRATCH:-/scratch/09281/chc_1996}/d3gym_images}
export D3GYM_WORKDIR_ROOT=${D3GYM_WORKDIR_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/d3gym_workdirs}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-${SCRATCH:-/scratch/09281/chc_1996}/apptainer_cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-${SCRATCH:-/scratch/09281/chc_1996}/apptainer_tmp}
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}

case "$D3GYM_METHOD" in
  react) DATA_SUFFIX=code ;;
  fold) DATA_SUFFIX=code_branch ;;
  ctxgraph) DATA_SUFFIX=code_graph ;;
  *) echo "ERROR: D3GYM_METHOD must be react, fold, or ctxgraph; got $D3GYM_METHOD"; exit 1 ;;
esac

export SCIENCE_BENCHMARK_LABEL=D3-Gym
export SCIENCE_TRAIN_FILE=data/d3gym_train_${DATA_SUFFIX}.parquet
export SCIENCE_VAL_FILE=data/d3gym_val_${DATA_SUFFIX}.parquet
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-d3gym_${D3GYM_METHOD}_30b_${D3GYM_MODE}_${SLURM_JOB_ID:-local}}
export SAB_METHOD=$D3GYM_METHOD
export SAB_REAL_EVAL=0
export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-8}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
export CONDA_ENV_NAME=${CONDA_ENV_NAME:-cxtgraph}
export SAB_PROMPT_LENGTH=${SAB_PROMPT_LENGTH:-16384}
export SAB_RESPONSE_LENGTH=${SAB_RESPONSE_LENGTH:-24576}
export SAB_MAX_TOKEN_LEN_PER_GPU=${SAB_MAX_TOKEN_LEN_PER_GPU:-40960}
export SAB_VAL_MAX_TURN=${SAB_VAL_MAX_TURN:-32}
export SAB_TURN_MAX_NEW_TOKENS=${SAB_TURN_MAX_NEW_TOKENS:-2048}
export SANDBOX_TIMEOUT=${SANDBOX_TIMEOUT:-300}
export EVAL_TIMEOUT=${EVAL_TIMEOUT:-480}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}

case "$D3GYM_MODE" in
  smoke)
    export SAB_RUN_TAG=d3gym_smoke
    export TRAINER_VAL_ONLY=True
    export TRAINER_VAL_BEFORE_TRAIN=True
    export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
    export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
    export ROLLOUT_N=${ROLLOUT_N:-1}
    export SAB_TRAIN_MAX_SAMPLES=${SAB_TRAIN_MAX_SAMPLES:-8}
    export SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:-1}
    export TOTAL_TRAINING_STEPS=1
    export TEST_FREQ=999
    export SAVE_FREQ=999
    export SAB_DISABLE_WANDB=${SAB_DISABLE_WANDB:-1}
    ;;
  eval)
    export SAB_RUN_TAG=d3gym_eval
    export TRAINER_VAL_ONLY=True
    export TRAINER_VAL_BEFORE_TRAIN=True
    export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
    export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
    export ROLLOUT_N=${ROLLOUT_N:-1}
    export SAB_TRAIN_MAX_SAMPLES=${SAB_TRAIN_MAX_SAMPLES:-8}
    export SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:--1}
    export TOTAL_TRAINING_STEPS=1
    export TEST_FREQ=999
    export SAVE_FREQ=999
    ;;
  train)
    export SAB_RUN_TAG=d3gym_train
    export TRAINER_VAL_ONLY=False
    export TRAINER_VAL_BEFORE_TRAIN=True
    export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
    export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
    export ROLLOUT_N=${ROLLOUT_N:-4}
    export SAB_TRAIN_MAX_SAMPLES=${SAB_TRAIN_MAX_SAMPLES:--1}
    export SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:-16}
    export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
    export TEST_FREQ=${TEST_FREQ:-10}
    export SAVE_FREQ=${SAVE_FREQ:-10}
    ;;
  *) echo "ERROR: D3GYM_MODE must be smoke, eval, or train; got $D3GYM_MODE"; exit 1 ;;
esac

if [ "$D3GYM_METHOD" = react ]; then
  export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
else
  export ADV_ESTIMATOR=${ADV_ESTIMATOR:-foldgrpo}
fi

if ! command -v "$D3GYM_RUNTIME" >/dev/null 2>&1; then
  echo "ERROR: D3GYM_RUNTIME=$D3GYM_RUNTIME is not available on $(hostname -s)"
  echo "       Load Apptainer or set D3GYM_RUNTIME=docker where Docker is available."
  exit 1
fi
mkdir -p "$D3GYM_IMAGE_DIR" "$D3GYM_WORKDIR_ROOT" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [ "$D3GYM_RUNTIME" = apptainer ] || [ "$D3GYM_RUNTIME" = singularity ]; then
  IMAGE_CHECK=(python scripts/cache_d3gym_images.py --check-only --check-arch --runtime "$D3GYM_RUNTIME" --image-dir "$D3GYM_IMAGE_DIR")
  case "$D3GYM_MODE" in
    smoke) IMAGE_CHECK+=(--limit "$SAB_VAL_MAX_SAMPLES" --parquet "$SCIENCE_VAL_FILE") ;;
    eval) IMAGE_CHECK+=(--parquet "$SCIENCE_VAL_FILE") ;;
    train) IMAGE_CHECK+=(--parquet "$SCIENCE_TRAIN_FILE" "$SCIENCE_VAL_FILE") ;;
  esac
  if ! "${IMAGE_CHECK[@]}"; then
    echo "Cache the required task images before submitting GPU work. Example:"
    echo "python scripts/cache_d3gym_images.py --parquet $SCIENCE_VAL_FILE --image-dir $D3GYM_IMAGE_DIR --runtime $D3GYM_RUNTIME"
    exit 1
  fi
fi

echo "D3-Gym: mode=$D3GYM_MODE method=$D3GYM_METHOD runtime=$D3GYM_RUNTIME image_dir=$D3GYM_IMAGE_DIR"
exec bash scripts/eval_sab_react_30b_instruct_8node_smoke.sh
