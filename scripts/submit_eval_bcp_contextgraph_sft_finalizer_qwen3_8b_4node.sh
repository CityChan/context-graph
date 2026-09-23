#!/bin/bash
#SBATCH -J bcp-sftfin8b
#SBATCH -o logs/bcp-sftfin8b.%j.out
#SBATCH -e logs/bcp-sftfin8b.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# Evaluate the merged ContextGraph Qwen3-8B SFT checkpoint on the complete
# BrowseComp-Plus test split using the current controller and shared finalizer.
# Structured memory is intentionally disabled so this run isolates the SFT
# checkpoint relative to the ContextGraph + Finalizer base-model control.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/09281/chc_1996}
SFT_EVAL_MODEL_PATH=${SFT_EVAL_MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/954050_qwen3_8b_contextgraph_sft_32k_fullparam}
EVAL_TIME=${EVAL_TIME:-04:00:00}
SCRIPT_PATH=$(readlink -f "$0")

cd "$PROJECT_ROOT"
mkdir -p logs

if [ -z "${SLURM_JOB_ID:-}" ]; then
  submission=$(sbatch --parsable --nodes=4 --time="$EVAL_TIME" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,SCRATCH_ROOT=$SCRATCH_ROOT,SFT_EVAL_MODEL_PATH=$SFT_EVAL_MODEL_PATH" "$SCRIPT_PATH")
  job_id=${submission%%;*}
  if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse job id from: $submission" >&2
    exit 3
  fi
  echo "Submitted BC-P ContextGraph SFT + Finalizer evaluation: $job_id"
  squeue -j "$job_id" -o '%.18i %.9P %.32j %.2t %.10M %.10L %.6D %R' || true
  exit 0
fi

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#ALLOC_NODES[@]}" -ne 4 ]; then
  echo "ERROR: expected exactly four allocated nodes, got ${#ALLOC_NODES[@]}" >&2
  exit 2
fi

if [ ! -s "$SFT_EVAL_MODEL_PATH/config.json" ]; then
  echo "ERROR: merged SFT model is missing config.json: $SFT_EVAL_MODEL_PATH" >&2
  exit 2
fi
if ! find "$SFT_EVAL_MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit | grep -q .; then
  echo "ERROR: merged SFT model has no non-empty Hugging Face weight files: $SFT_EVAL_MODEL_PATH" >&2
  exit 2
fi

export EVAL_MODEL_PATH="$SFT_EVAL_MODEL_PATH"
export EVAL_MODEL_TAG=qwen3_8b_contextgraph_sft_step592
export EVAL_VARIANT=contextgraph_sft_finalizer_bcp
export EVAL_CTXGRAPH_PROTOCOL=controller
export RUN_BC=1
export RUN_GAIA=0

export BC_CONTEXT_LENGTH=65536
export BC_PROMPT_LENGTH=8192
export BC_RESPONSE_LENGTH=57344
export BC_MAX_TURN=100
export BC_MAX_SESSION=10
export BC_FINAL_ANSWER_RESERVE=1024

export STRUCTURED_MEMORY_ENABLED=0
export STRUCTURED_MEMORY_REQUIRED=0
export BC_DISABLE_WANDB=1
export WANDB_MODE=disabled

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_LOG=${RUN_LOG:-logs/bcp-sftfin8b.${SLURM_JOB_ID}.${STAMP}.log}
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=============================================================="
echo "  BC-P: ContextGraph SFT + Finalizer"
echo "  Job: $SLURM_JOB_ID"
echo "  Log: $RUN_LOG"
echo "  Model: $EVAL_MODEL_PATH"
echo "  Nodes: ${ALLOC_NODES[*]}"
echo "  Samples: complete BC-P test split"
echo "  Context: $BC_CONTEXT_LENGTH; max_turn: $BC_MAX_TURN"
echo "  Finalizer reserve: $BC_FINAL_ANSWER_RESERVE"
echo "  StructMem: disabled"
echo "=============================================================="

set +e
bash scripts/eval_bc_gaia_qwen3_8b_base_idev.sh
rc=$?
set -e

echo "=============================================================="
echo "  BC-P ContextGraph SFT + Finalizer run finished with exit code $rc"
echo "  Full log: $RUN_LOG"
echo "=============================================================="
exit "$rc"
