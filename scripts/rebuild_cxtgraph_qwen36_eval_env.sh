#!/bin/bash
# Rebuild the formal cxtgraph eval environment with the validated Qwen3.6 stack.
# The old environment is retained under BACKUP_ENV for rollback.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
CONDA_ROOT=${CONDA_ROOT:-/work/09281/chc_1996/vista/miniconda3}
SOURCE_ENV=${SOURCE_ENV:-deepseek_v4}
TARGET_ENV=${TARGET_ENV:-cxtgraph}
STAGE_ENV=${STAGE_ENV:-cxtgraph_qwen36_stage_20260830}
BACKUP_ENV=${BACKUP_ENV:-cxtgraph_pre_qwen36_20260830}
MODEL_PATH=${MODEL_PATH:-${SCRATCH:-/scratch/09281/chc_1996}/hf_cache/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}

if [ "${HOSTNAME%%.*}" != login1 ] && [ "${HOSTNAME%%.*}" != login2 ]; then
  echo "ERROR: run this environment migration on a Vista login node" >&2
  exit 2
fi
if [ -n "$(squeue -h -u "${USER:-$(whoami)}")" ]; then
  echo "ERROR: active or queued Slurm jobs exist; refusing to replace $TARGET_ENV" >&2
  squeue -u "${USER:-$(whoami)}"
  exit 2
fi
if [ ! -s "$MODEL_PATH/config.json" ]; then
  echo "ERROR: invalid MODEL_PATH: $MODEL_PATH" >&2
  exit 2
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda deactivate >/dev/null 2>&1 || true

env_exists() {
  conda env list | awk '{print $1}' | grep -Fxq "$1"
}

for required_env in "$SOURCE_ENV" "$TARGET_ENV"; do
  if ! env_exists "$required_env"; then
    echo "ERROR: required conda environment does not exist: $required_env" >&2
    exit 2
  fi
done
for new_env in "$STAGE_ENV" "$BACKUP_ENV"; do
  if env_exists "$new_env"; then
    echo "ERROR: rollback/staging environment already exists: $new_env" >&2
    exit 2
  fi
done

preflight_env() {
  local env_name=$1
  echo "+++ preflighting $env_name"
  conda run -n "$env_name" python -c "import datasets,fastapi,httpx,numpy,pandas,pyarrow,torch,transformers,uvicorn,vllm,wandb; from transformers import AutoConfig; from vllm.sampling_params import StructuredOutputsParams; c=AutoConfig.from_pretrained('$MODEL_PATH',local_files_only=True); assert getattr(c,'model_type',None)=='qwen3_5'; assert hasattr(wandb,'init'); print('env preflight:',transformers.__version__,vllm.__version__,torch.__version__,wandb.__version__)"
  MODEL_PATH="$MODEL_PATH" TORCHDYNAMO_DISABLE=0 VLLM_USE_AOT_COMPILE=1 conda run -n "$env_name" python scripts/check_vllm_structured_outputs.py
  MODEL_PATH="$MODEL_PATH" TORCHDYNAMO_DISABLE=0 VLLM_USE_AOT_COMPILE=1 conda run -n "$env_name" python scripts/check_vllm_eval_compat.py
}

cd "$PROJECT_ROOT"
echo "+++ cloning $SOURCE_ENV -> $STAGE_ENV"
conda create -y -n "$STAGE_ENV" --clone "$SOURCE_ENV"
conda run -n "$STAGE_ENV" python -m pip install --upgrade wandb==0.25.1
preflight_env "$STAGE_ENV"

echo "+++ preserving $TARGET_ENV -> $BACKUP_ENV"
conda create -y -n "$BACKUP_ENV" --clone "$TARGET_ENV"
echo "+++ replacing $TARGET_ENV from validated staging environment"
conda env remove -y -n "$TARGET_ENV"
conda create -y -n "$TARGET_ENV" --clone "$STAGE_ENV"
preflight_env "$TARGET_ENV"

echo "=============================================================="
echo "  cxtgraph Qwen3.6 eval environment migration completed"
echo "  Active:   $TARGET_ENV"
echo "  Rollback: $BACKUP_ENV"
echo "  Staging:  $STAGE_ENV"
echo "=============================================================="
