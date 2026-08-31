#!/bin/bash

# Submit protocol-matched ContextGraph evaluations of the merged Qwen3-8B SFT
# model on BrowseComp-Plus, GAIA, and DiscoveryBench.
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_PATH=${SFT_EVAL_MODEL_PATH:-$SCRATCH/contextgraph_sft_models/954050_qwen3_8b_contextgraph_sft_32k_fullparam}
HF_HOME=/work/09281/chc_1996/vista/cache
HF_HUB_CACHE=$HF_HOME/hub
DRY_RUN=${DRY_RUN:-0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

case "$DRY_RUN" in
  0|1) ;;
  *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac
if [ ! -s "$MODEL_PATH/config.json" ]; then
  echo "ERROR: merged SFT model is missing config.json: $MODEL_PATH" >&2
  exit 2
fi
if ! find "$MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit | grep -q .; then
  echo "ERROR: merged SFT model has no non-empty Hugging Face weight files: $MODEL_PATH" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p logs

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
    return
  fi
  output=$("${command[@]}")
  job_id=$(printf '%s\n' "$output" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$job_id" ]; then
    echo "ERROR: could not parse $benchmark job id from: $output" >&2
    exit 3
  fi
  printf '%-18s job=%s\n' "$benchmark" "$job_id"
}

BC_EXPERIMENT=eval_contextgraph_controller_bc_qwen3_8b_sft_32k_4n_64k_${STAMP}
submit_job browsecomp sbatch --parsable --job-name=eval-bc-sft8b-ctxgraph-controller-64k --nodes=4 --time=02:00:00 --output="logs/eval-bc-sft8b-ctxgraph-controller-64k.%j.out" --error="logs/eval-bc-sft8b-ctxgraph-controller-64k.%j.err" --export="ALL,MODEL_PATH=$MODEL_PATH,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,BC_METHOD=contextgraph,BC_CTXGRAPH_PROTOCOL=controller,BC_CONTROLLER_ACTION_POLICY=structural,BC_EXPERIMENT_MODEL_TAG=qwen3_8b_sft_32k,BC_CONTEXT_LENGTH=65536,BC_PROMPT_LENGTH=8192,BC_RESPONSE_LENGTH=57344,BC_YARN_FACTOR=2.0,BC_YARN_ORIGINAL_LENGTH=32768,BC_FINAL_ANSWER_RESERVE=1024,BC_VAL_MAX_SAMPLES=-1,BC_DISABLE_WANDB=1,EXPERIMENT_NAME=$BC_EXPERIMENT" scripts/eval_bc_baseline_8b_4node_zeroshot.sh

GAIA_EXPERIMENT=eval_gaia_ctxgraph_controller_qwen3_8b_sft_32k_zeroshot_${STAMP}
GAIA_OUTPUT_ROOT="$SCRATCH/context-graph-ckpts/$GAIA_EXPERIMENT"
submit_job gaia sbatch --parsable --job-name=gaia-ctxgraph-controller-sft8b-32k --nodes=5 --time=03:00:00 --output="logs/gaia-ctxgraph-controller-sft8b-32k.%j.out" --error="logs/gaia-ctxgraph-controller-sft8b-32k.%j.err" --export="ALL,MODEL_PATH=$MODEL_PATH,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,EXPECTED_NUM_NODES=5,EXPERIMENT_NAME=$GAIA_EXPERIMENT,CHECKPOINT_ROOT=$GAIA_OUTPUT_ROOT,TRAIN_DATA_FILE=data/gaia_validation_graph.parquet,VAL_DATA_FILE=data/gaia_validation_graph.parquet,TRAIN_MAX_SAMPLES=-1,VAL_MAX_SAMPLES=-1,TRAINER_VAL_ONLY=True,VAL_BEFORE_TRAIN=True,TOTAL_TRAINING_STEPS=1,TEST_FREQ=999,SAVE_FREQ=-1,PROMPT_LENGTH=8192,RESPONSE_LENGTH=24576,CONTEXT_LENGTH=32768,TRAIN_BATCH_SIZE=32,PPO_MINI_BATCH_SIZE=16,ROLLOUT_N=1,MAX_TURN=100,MAX_SESSION=10,VAL_MAX_SESSION=10,TURN_MAX_NEW_TOKENS=2048,FINAL_ANSWER_RESERVE=1024,SESSION_TIMEOUT=3600,BC_SEARCH_TIMEOUT_SECONDS=600,BC_CTXGRAPH_PROTOCOL=controller,BC_CONTROLLER_ACTION_POLICY=balanced,BC_DISABLE_WANDB=1" scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh

DB_EXPERIMENT=eval_ctxgraph_controller_discoverybench_qwen3_8b_sft_32k_4n_${STAMP}
submit_job discoverybench sbatch --parsable --job-name=eval-db-ctxgraph-controller-sft8b-4n --nodes=4 --time=08:00:00 --output="logs/eval-db-ctxgraph-controller-sft8b-4n.%j.out" --error="logs/eval-db-ctxgraph-controller-sft8b-4n.%j.err" --export="ALL,MODEL_PATH=$MODEL_PATH,HF_HOME=$HF_HOME,HF_HUB_CACHE=$HF_HUB_CACHE,CONDA_ENV_NAME=cxtgraph,DISCOVERYBENCH_METHOD=ctxgraph,SAB_CTXGRAPH_PROTOCOL=controller,SAB_CONTROLLER_ACTION_POLICY=structural,DISCOVERYBENCH_VAL_MAX_SAMPLES=239,DISCOVERYBENCH_TRAIN_MAX_SAMPLES=4,DISCOVERYBENCH_RESPONSE_LENGTH=8192,DISCOVERYBENCH_MAX_TOKEN_LEN_PER_GPU=24576,SAB_DISABLE_WANDB=1,EXPERIMENT_NAME=$DB_EXPERIMENT" scripts/eval_discoverybench_qwen3_8b_4node.sh

if [ "$DRY_RUN" = "0" ]; then
  squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.38j %.2t %.10M %.10L %.6D %R"
fi
