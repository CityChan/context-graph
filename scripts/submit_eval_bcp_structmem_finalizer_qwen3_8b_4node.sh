#!/bin/bash
#SBATCH -J bcp-smfin8b
#SBATCH -o logs/bcp-smfin8b.%j.out
#SBATCH -e logs/bcp-smfin8b.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# Run the missing paper-table treatment on the complete BC-P test split:
# ContextGraph + StructMem + the shared 1024-token emergency finalizer.
#
# The script self-submits from a login node and runs directly inside an
# existing four-node idev allocation. The explicit tee log is required because
# #SBATCH stdout/stderr files are not reliable for direct idev execution.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
EVAL_TIME=${EVAL_TIME:-04:00:00}
SCRIPT_PATH=$(readlink -f "$0")

cd "$PROJECT_ROOT"
mkdir -p logs

if [ -z "${SLURM_JOB_ID:-}" ]; then
  submission=$(sbatch --parsable --nodes=4 --time="$EVAL_TIME" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT" "$SCRIPT_PATH")
  job_id=${submission%%;*}
  if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse job id from: $submission" >&2
    exit 3
  fi
  echo "Submitted BC-P ContextGraph + StructMem + Finalizer evaluation: $job_id"
  squeue -j "$job_id" -o '%.18i %.9P %.32j %.2t %.10M %.10L %.6D %R' || true
  exit 0
fi

mapfile -t ALLOC_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#ALLOC_NODES[@]}" -ne 4 ]; then
  echo "ERROR: expected exactly four allocated nodes, got ${#ALLOC_NODES[@]}" >&2
  exit 2
fi

export EVAL_MODEL_PATH=${EVAL_MODEL_PATH:-Qwen/Qwen3-8B}
export EVAL_MODEL_TAG=${EVAL_MODEL_TAG:-qwen3_8b_base}
export EVAL_VARIANT=${EVAL_VARIANT:-base_structmem_finalizer_fixed_bcp}
export EVAL_CTXGRAPH_PROTOCOL=controller
export RUN_BC=1
export RUN_GAIA=0

export BC_CONTEXT_LENGTH=65536
export BC_PROMPT_LENGTH=8192
export BC_RESPONSE_LENGTH=57344
export BC_MAX_TURN=100
export BC_MAX_SESSION=10
export BC_FINAL_ANSWER_RESERVE=1024

export STRUCTURED_MEMORY_ENABLED=1
export STRUCTURED_MEMORY_REQUIRED=1
export STRUCTURED_MEMORY_GAP_INTERVAL=${STRUCTURED_MEMORY_GAP_INTERVAL:-8}
export STRUCTURED_MEMORY_CONTEXT_BUDGET=${STRUCTURED_MEMORY_CONTEXT_BUDGET:-1024}
export STRUCTURED_MEMORY_MAX_CONTEXT_FACTS=${STRUCTURED_MEMORY_MAX_CONTEXT_FACTS:-12}
export STRUCTURED_MEMORY_MAX_FACTS_PER_OBSERVATION=${STRUCTURED_MEMORY_MAX_FACTS_PER_OBSERVATION:-8}
export STRUCTURED_MEMORY_MAX_FACTS=${STRUCTURED_MEMORY_MAX_FACTS:-128}
export STRUCTURED_MEMORY_EXTRACT_MAX_TOKENS=${STRUCTURED_MEMORY_EXTRACT_MAX_TOKENS:-768}
export STRUCTURED_MEMORY_GAP_MAX_TOKENS=${STRUCTURED_MEMORY_GAP_MAX_TOKENS:-512}
export STRUCTURED_MEMORY_PLAN_MAX_TOKENS=${STRUCTURED_MEMORY_PLAN_MAX_TOKENS:-512}
export STRUCTURED_MEMORY_RELATION_CANDIDATES=${STRUCTURED_MEMORY_RELATION_CANDIDATES:-128}
export STRUCTURED_MEMORY_CONTROLLER_RETRIES=${STRUCTURED_MEMORY_CONTROLLER_RETRIES:-2}
export STRUCTURED_MEMORY_GAP_JITTER=${STRUCTURED_MEMORY_GAP_JITTER:-1}
export STRUCTURED_MEMORY_STOP_ON_READY=1
export STRUCTURED_MEMORY_STEP_LIMIT=${STRUCTURED_MEMORY_STEP_LIMIT:-40}

export BC_DISABLE_WANDB=1
export WANDB_MODE=disabled

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_LOG=${RUN_LOG:-logs/bcp-smfin8b.${SLURM_JOB_ID}.${STAMP}.log}
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=============================================================="
echo "  BC-P: ContextGraph + StructMem + Finalizer"
echo "  Job: $SLURM_JOB_ID"
echo "  Log: $RUN_LOG"
echo "  Model: $EVAL_MODEL_PATH"
echo "  Nodes: ${ALLOC_NODES[*]}"
echo "  Samples: complete BC-P test split"
echo "  Context: $BC_CONTEXT_LENGTH; max_turn: $BC_MAX_TURN"
echo "  Finalizer reserve: $BC_FINAL_ANSWER_RESERVE"
echo "  StructMem: enabled=$STRUCTURED_MEMORY_ENABLED required=$STRUCTURED_MEMORY_REQUIRED stop_on_ready=$STRUCTURED_MEMORY_STOP_ON_READY step_limit=$STRUCTURED_MEMORY_STEP_LIMIT"
echo "=============================================================="

set +e
bash scripts/eval_bc_gaia_qwen3_8b_base_idev.sh
rc=$?
set -e

echo "=============================================================="
echo "  BC-P StructMem + Finalizer run finished with exit code $rc"
echo "  Full log: $RUN_LOG"
echo "=============================================================="
exit "$rc"
