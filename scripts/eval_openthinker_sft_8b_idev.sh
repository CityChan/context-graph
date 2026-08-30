#!/bin/bash
# Run one OpenThinker 8B SFT ContextGraph evaluation in an active 4-node idev allocation.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_CACHE_ROOT=${MODEL_CACHE_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/hf_cache/hub/models--open-thoughts--OpenThinkerAgent-8B-ColdStartSFTForRL}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-1}
CONTROLLER_ACTION_POLICY=${CONTROLLER_ACTION_POLICY:-balanced}

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this script inside an active four-node Vista idev allocation"
  exit 1
fi
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#NODELIST[@]}" -ne 4 ]; then
  echo "ERROR: expected exactly four idev nodes, got ${#NODELIST[@]}"
  exit 1
fi
if ! [[ "$EVAL_MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EVAL_MAX_SAMPLES must be a positive integer"
  exit 1
fi

if [ -z "${MODEL_PATH:-}" ]; then
  shopt -s nullglob
  candidates=("$MODEL_CACHE_ROOT"/snapshots/*)
  shopt -u nullglob
  for candidate in "${candidates[@]}"; do
    if [ -s "$candidate/config.json" ] && [ -s "$candidate/model.safetensors.index.json" ]; then
      MODEL_PATH=$candidate
      break
    fi
  done
fi
if [ -z "${MODEL_PATH:-}" ] || [ ! -s "$MODEL_PATH/config.json" ] || [ ! -s "$MODEL_PATH/model.safetensors.index.json" ]; then
  echo "ERROR: OpenThinker 8B SFT snapshot not found under $MODEL_CACHE_ROOT"
  echo "Run the download-only submitter on a Vista login node first."
  exit 2
fi

cd "$PROJECT_ROOT"
unset LORA_ADAPTER_PATH LORA_RANK LORA_ALPHA
export MODEL_PATH
export CONDA_ENV_NAME=deepseek_v4 SEARCH_CONDA_ENV_NAME=cxtgraph
export EXPECTED_NUM_NODES=4 BC_METHOD=contextgraph BC_CTXGRAPH_PROTOCOL=controller
export BC_CONTROLLER_ACTION_POLICY=$CONTROLLER_ACTION_POLICY BC_EXPERIMENT_MODEL_TAG=openthinker_sft_8b
export BC_VAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES BC_ROLLOUT_N=1 BC_DISABLE_WANDB=1
export BC_CONTEXT_LENGTH=32768 BC_PROMPT_LENGTH=8192 BC_RESPONSE_LENGTH=24576
export BC_MAX_TOKEN_LEN_PER_GPU=32768 BC_FINAL_ANSWER_RESERVE=1024
export BC_ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1 BC_ROLLOUT_GPU_MEMORY_UTILIZATION=0.6 BC_TRAINER_NNODES=3
export ROLLOUT_LOAD_FORMAT=dummy ROLLOUT_LAYERED_SUMMON=False
export QWEN_ENABLE_THINKING=True TORCHDYNAMO_DISABLE=0 VLLM_USE_AOT_COMPILE=1

echo "OpenThinker 8B SFT idev eval: model=$MODEL_PATH samples=$EVAL_MAX_SAMPLES protocol=controller policy=$CONTROLLER_ACTION_POLICY nodes=${NODELIST[*]}"
exec bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh
