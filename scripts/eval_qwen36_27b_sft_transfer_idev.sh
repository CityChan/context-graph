#!/bin/bash
# Run one Qwen3.6-27B ContextGraph transfer eval directly in an active idev allocation.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-${SCRATCH:-/scratch/09281/chc_1996}/hf_cache/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-${SCRATCH:-/scratch/09281/chc_1996}/contextgraph_sft_exports/qwen36_27b_scienceworld_step147/lora_adapter}
BENCHMARK=${BENCHMARK:-bc}
VARIANT=${VARIANT:-base}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-1}
CONTROLLER_ACTION_POLICY=${CONTROLLER_ACTION_POLICY:-structural}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-deepseek_v4}
SEARCH_CONDA_ENV_NAME=${SEARCH_CONDA_ENV_NAME:-cxtgraph}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this script inside an active idev allocation"
  exit 2
fi
if ! [[ "$EVAL_MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EVAL_MAX_SAMPLES must be a positive integer for idev smoke"
  exit 2
fi
case "$BENCHMARK" in gaia|bc|discovery) ;; *) echo "ERROR: BENCHMARK must be gaia, bc, or discovery"; exit 2 ;; esac
case "$VARIANT" in base|sft) ;; *) echo "ERROR: VARIANT must be base or sft"; exit 2 ;; esac
case "$CONTROLLER_ACTION_POLICY" in balanced|structural) ;; *) echo "ERROR: CONTROLLER_ACTION_POLICY must be balanced or structural"; exit 2 ;; esac

mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NUM_NODES=${#NODELIST[@]}
REQUIRED_NODES=4
if [ "$BENCHMARK" = gaia ]; then REQUIRED_NODES=5; fi
if [ "$NUM_NODES" -ne "$REQUIRED_NODES" ]; then
  echo "ERROR: $BENCHMARK requires exactly $REQUIRED_NODES nodes; allocation has $NUM_NODES (${NODELIST[*]})"
  exit 2
fi
if [ ! -s "$BASE_MODEL_PATH/config.json" ]; then
  echo "ERROR: invalid BASE_MODEL_PATH: $BASE_MODEL_PATH"
  exit 2
fi

adapter_path=
adapter_rank=0
if [ "$VARIANT" = sft ]; then
  for required in adapter_config.json adapter_model.safetensors; do
    if [ ! -s "$LORA_ADAPTER_PATH/$required" ]; then
      echo "ERROR: missing adapter file: $LORA_ADAPTER_PATH/$required"
      exit 2
    fi
  done
  adapter_path=$LORA_ADAPTER_PATH
  adapter_rank=$LORA_RANK
fi

cd "$PROJECT_ROOT"
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache_qwen36_idev_${SLURM_JOB_ID:-local}_${BENCHMARK}_${VARIANT}}
export MODEL_PATH=$BASE_MODEL_PATH
export CONDA_ENV_NAME SEARCH_CONDA_ENV_NAME
export LORA_ADAPTER_PATH=$adapter_path LORA_RANK=$adapter_rank LORA_ALPHA
export QWEN_ENABLE_THINKING=True
# Qwen3.5/3.6 uses vLLM's AOT torch.compile path during the profile run.
# Legacy benchmark runners default to eager execution for older backbones, so
# explicitly override that default for this transfer evaluation.
export TORCHDYNAMO_DISABLE=0
export VLLM_USE_AOT_COMPILE=1

# Qwen3.6-27B's hybrid GDN/full-attention cache needs about 98 GiB at
# TP=1 for the unchanged 32K protocol. TP=2 halves both model and cache
# residency per GPU; the higher utilization leaves enough room for that cache.

echo "Qwen3.6-27B idev eval: benchmark=$BENCHMARK variant=$VARIANT samples=$EVAL_MAX_SAMPLES protocol=controller policy=$CONTROLLER_ACTION_POLICY nodes=${NODELIST[*]}"

case "$BENCHMARK" in
  bc)
    export EXPECTED_NUM_NODES=4 BC_METHOD=contextgraph BC_CTXGRAPH_PROTOCOL=controller BC_CONTROLLER_ACTION_POLICY=$CONTROLLER_ACTION_POLICY
    export BC_EXPERIMENT_MODEL_TAG=qwen36_27b_${VARIANT} BC_VAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES BC_ROLLOUT_N=1
    export BC_ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2 BC_ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 BC_TRAINER_NNODES=2
    export BC_PROMPT_LENGTH=8192 BC_RESPONSE_LENGTH=24576 BC_CONTEXT_LENGTH=32768 BC_FINAL_ANSWER_RESERVE=1024
    export EXPERIMENT_NAME=idev_eval_bc_contextgraph_controller_qwen36_27b_${VARIANT}_${STAMP}
    exec bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh
    ;;
  gaia)
    export EXPECTED_NUM_NODES=5 BC_CTXGRAPH_PROTOCOL=controller BC_CONTROLLER_ACTION_POLICY=$CONTROLLER_ACTION_POLICY
    export TRAIN_DATA_FILE=data/gaia_validation_graph.parquet VAL_DATA_FILE=data/gaia_validation_graph.parquet
    export TRAIN_MAX_SAMPLES=$EVAL_MAX_SAMPLES VAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES TRAIN_BATCH_SIZE=$EVAL_MAX_SAMPLES PPO_MINI_BATCH_SIZE=$EVAL_MAX_SAMPLES
    export TRAINER_VAL_ONLY=True VAL_BEFORE_TRAIN=True TOTAL_TRAINING_STEPS=1 TEST_FREQ=999 SAVE_FREQ=-1 ROLLOUT_N=1
    export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2 ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
    export PROMPT_LENGTH=8192 RESPONSE_LENGTH=24576 CONTEXT_LENGTH=32768 FINAL_ANSWER_RESERVE=1024
    export MAX_TURN=100 MAX_SESSION=10 VAL_MAX_SESSION=10 TURN_MAX_NEW_TOKENS=2048 SESSION_TIMEOUT=3600
    export EXPERIMENT_NAME=idev_eval_gaia_ctxgraph_controller_qwen36_27b_${VARIANT}_${STAMP}
    export CHECKPOINT_ROOT=${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/$EXPERIMENT_NAME
    exec bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
    ;;
  discovery)
    export EXPECTED_NUM_NODES=4 DISCOVERYBENCH_METHOD=ctxgraph SAB_CTXGRAPH_PROTOCOL=controller SAB_CONTROLLER_ACTION_POLICY=$CONTROLLER_ACTION_POLICY
    export DISCOVERYBENCH_VAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES DISCOVERYBENCH_TRAIN_MAX_SAMPLES=4
    export SAB_ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2 SAB_ROLLOUT_GPU_MEMORY_UTILIZATION=0.9 SAB_ROLLOUT_QUANTIZATION=none
    export DISCOVERYBENCH_PROMPT_LENGTH=16384 DISCOVERYBENCH_RESPONSE_LENGTH=16384 DISCOVERYBENCH_MAX_TOKEN_LEN_PER_GPU=32768
    export DISCOVERYBENCH_VAL_MAX_TURN=32 DISCOVERYBENCH_TURN_MAX_NEW_TOKENS=2048 SAB_RUN_TAG=idev
    export DISCOVERYBENCH_EXPERIMENT_MODEL_TAG=qwen36_27b_${VARIANT}
    export EXPERIMENT_NAME=idev_eval_discovery_ctxgraph_controller_qwen36_27b_${VARIANT}_${STAMP}
    exec bash scripts/eval_discoverybench_qwen3_8b_4node.sh
    ;;
esac
