#!/bin/bash
# Submit matched ContextGraph before/after-SFT transfer evaluations.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-${SCRATCH:-/scratch/09281/chc_1996}/hf_cache/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-${SCRATCH:-/scratch/09281/chc_1996}/contextgraph_sft_exports/qwen36_27b_scienceworld_step147/lora_adapter}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
VARIANTS=${VARIANTS:-base,sft}
BENCHMARKS=${BENCHMARKS:-gaia,bc,discovery}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:--1}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-deepseek_v4}
SEARCH_CONDA_ENV_NAME=${SEARCH_CONDA_ENV_NAME:-cxtgraph}
DRY_RUN=${DRY_RUN:-0}

if ! [[ "$EVAL_MAX_SAMPLES" =~ ^(-1|[1-9][0-9]*)$ ]]; then
  echo "ERROR: EVAL_MAX_SAMPLES must be -1 or a positive integer" >&2
  exit 2
fi
if [ "$EVAL_MAX_SAMPLES" = "1" ]; then
  DEFAULT_EVAL_TIME=00:30:00
else
  DEFAULT_EVAL_TIME=01:30:00
fi
GAIA_EVAL_TIME=${GAIA_EVAL_TIME:-$DEFAULT_EVAL_TIME}
BC_EVAL_TIME=${BC_EVAL_TIME:-$DEFAULT_EVAL_TIME}
DISCOVERYBENCH_TIME_LIMIT=${DISCOVERYBENCH_TIME_LIMIT:-$DEFAULT_EVAL_TIME}
DISCOVERYBENCH_VAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES
if [ "$DISCOVERYBENCH_VAL_MAX_SAMPLES" = "-1" ]; then
  DISCOVERYBENCH_VAL_MAX_SAMPLES=239
fi

if [ ! -s "$BASE_MODEL_PATH/config.json" ]; then
  echo "ERROR: invalid BASE_MODEL_PATH: $BASE_MODEL_PATH" >&2
  exit 2
fi
for required in adapter_config.json adapter_model.safetensors; do
  if [ ! -s "$LORA_ADAPTER_PATH/$required" ]; then
    echo "ERROR: missing adapter file: $LORA_ADAPTER_PATH/$required" >&2
    echo "Run scripts/export_contextgraph_sft_lora.sh on a compute node first." >&2
    exit 2
  fi
done
python -c 'import json,sys; config=json.load(open(sys.argv[1],encoding="utf-8")); expected_rank,expected_alpha=int(sys.argv[2]),int(sys.argv[3]); assert int(config.get("r",0))==expected_rank,(config,expected_rank); assert int(config.get("lora_alpha",0))==expected_alpha,(config,expected_alpha)' "$LORA_ADAPTER_PATH/adapter_config.json" "$LORA_RANK" "$LORA_ALPHA"

enabled() {
  local list=$1
  local item=$2
  case ",$list," in
    *",$item,"*) return 0 ;;
    *) return 1 ;;
  esac
}

cd "$PROJECT_ROOT"

for variant in base sft; do
  if ! enabled "$VARIANTS" "$variant"; then
    continue
  fi

  adapter_path=
  adapter_rank=0
  if [ "$variant" = "sft" ]; then
    adapter_path=$LORA_ADAPTER_PATH
    adapter_rank=$LORA_RANK
  fi
  job_tag="qwen36-27b-${variant}"
  experiment_tag="qwen36_27b_${variant}"
  common_env=(MODEL_PATH="$BASE_MODEL_PATH" CONDA_ENV_NAME="$CONDA_ENV_NAME" SEARCH_CONDA_ENV_NAME="$SEARCH_CONDA_ENV_NAME" LORA_ADAPTER_PATH="$adapter_path" LORA_RANK="$adapter_rank" LORA_ALPHA="$LORA_ALPHA" QWEN_ENABLE_THINKING=True DRY_RUN="$DRY_RUN")

  if enabled "$BENCHMARKS" gaia; then
    env "${common_env[@]}" GAIA_MODEL_PATH="$BASE_MODEL_PATH" GAIA_METHODS=ctxgraph GAIA_JOB_MODEL_TAG="$job_tag" GAIA_EXPERIMENT_MODEL_TAG="$experiment_tag" GAIA_CTXGRAPH_PROTOCOL=controller GAIA_CONTROLLER_ACTION_POLICY=balanced GAIA_VAL_MAX_SAMPLES="$EVAL_MAX_SAMPLES" GAIA_NUM_NODES=5 GAIA_EVAL_TIME="$GAIA_EVAL_TIME" PROMPT_LENGTH=8192 RESPONSE_LENGTH=24576 CONTEXT_LENGTH=32768 FINAL_ANSWER_RESERVE=1024 bash scripts/submit_gaia_benchmark_8b_5node.sh
  fi

  if enabled "$BENCHMARKS" bc; then
    env "${common_env[@]}" BC_METHODS=contextgraph BC_JOB_MODEL_TAG="$job_tag" BC_EXPERIMENT_MODEL_TAG="$experiment_tag" BC_CTXGRAPH_PROTOCOL=controller BC_CONTROLLER_ACTION_POLICY=balanced BC_VAL_MAX_SAMPLES="$EVAL_MAX_SAMPLES" BC_EVAL_TIME="$BC_EVAL_TIME" BC_PROMPT_LENGTH=8192 BC_RESPONSE_LENGTH=24576 BC_CONTEXT_LENGTH=32768 BC_FINAL_ANSWER_RESERVE=1024 bash scripts/submit_eval_bc_8b_4node_zeroshot_64k.sh
  fi

  if enabled "$BENCHMARKS" discovery; then
    env "${common_env[@]}" DISCOVERYBENCH_METHODS=ctxgraph DISCOVERYBENCH_JOB_MODEL_TAG="$job_tag" DISCOVERYBENCH_EXPERIMENT_MODEL_TAG="$experiment_tag" DISCOVERYBENCH_CTXGRAPH_PROTOCOL=controller DISCOVERYBENCH_CONTROLLER_ACTION_POLICY=balanced DISCOVERYBENCH_VAL_MAX_SAMPLES="$DISCOVERYBENCH_VAL_MAX_SAMPLES" DISCOVERYBENCH_TIME_LIMIT="$DISCOVERYBENCH_TIME_LIMIT" DISCOVERYBENCH_PROMPT_LENGTH=16384 DISCOVERYBENCH_RESPONSE_LENGTH=16384 DISCOVERYBENCH_MAX_TOKEN_LEN_PER_GPU=32768 DISCOVERYBENCH_VAL_MAX_TURN=32 DISCOVERYBENCH_TURN_MAX_NEW_TOKENS=2048 bash scripts/submit_eval_discoverybench_qwen3_8b_4node.sh
  fi
done

if [ "$DRY_RUN" = "0" ]; then
  squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.40j %.2t %.10M %.10L %.6D %R"
fi
