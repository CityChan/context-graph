#!/bin/bash
#SBATCH -J eval3-sm8b
#SBATCH -o logs/eval3-sm8b.%j.out
#SBATCH -e logs/eval3-sm8b.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 08:00:00
#SBATCH -A AST24021

# Submit or run a sequential paper-aligned StructMem evaluation on BC-P, GAIA,
# and ALFWorld using four Vista GH200 nodes and the Qwen3-8B base model.
# Invoking with bash on a login node self-submits. Inside an existing four-node
# allocation, invoking with bash runs the suite directly.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
EVAL_TIME=${EVAL_TIME:-08:00:00}
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
  echo "Submitted four-node BC-P + GAIA + ALFWorld StructMem evaluation: $job_id"
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
export EVAL_VARIANT=${EVAL_VARIANT:-base_structmem_v3_suite}
export EVAL_CTXGRAPH_PROTOCOL=controller
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
export ALFWORLD_DISABLE_WANDB=1
export WANDB_MODE=disabled

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
echo "=============================================================="
echo "  BC-P + GAIA + ALFWorld StructMem evaluation"
echo "  Job: $SLURM_JOB_ID"
echo "  Model: $EVAL_MODEL_PATH"
echo "  Nodes: ${ALLOC_NODES[*]}"
echo "  StructMem: gap=${STRUCTURED_MEMORY_GAP_INTERVAL}+/-${STRUCTURED_MEMORY_GAP_JITTER}, retries=${STRUCTURED_MEMORY_CONTROLLER_RETRIES}, facts=${STRUCTURED_MEMORY_MAX_FACTS}, step_limit=${STRUCTURED_MEMORY_STEP_LIMIT}"
echo "=============================================================="

set +e
RUN_BC=1 RUN_GAIA=1 STAMP="$STAMP" bash scripts/eval_bc_gaia_qwen3_8b_base_idev.sh
BC_GAIA_RC=$?
echo "BC-P + GAIA stage finished with exit code $BC_GAIA_RC"

# ALFWorld uses action observations for extraction and relies on the environment
# terminal signal, so can_answer early stopping remains disabled for this stage.
MODEL_PATH="$EVAL_MODEL_PATH" \
ALFWORLD_EXPERIMENT_PREFIX="${EVAL_VARIANT}_qwen3_8b" \
ALFWORLD_EVAL_SAMPLES="${ALFWORLD_EVAL_SAMPLES:-32}" \
ALFWORLD_STRUCTURED_MEMORY_ENABLED=1 \
ALFWORLD_STRUCTURED_MEMORY_REQUIRED=1 \
ALFWORLD_STRUCTURED_MEMORY_TOOLS=action,branch_return \
ALFWORLD_STRUCTURED_MEMORY_STOP_ON_READY=0 \
ALFWORLD_STRUCTURED_MEMORY_STEP_LIMIT="$STRUCTURED_MEMORY_STEP_LIMIT" \
bash scripts/eval_alfworld_ctxgraph_8b_4node_zeroshot_idev.sh
ALFWORLD_RC=$?
echo "ALFWorld stage finished with exit code $ALFWORLD_RC"
set -e

echo "=============================================================="
echo "  Evaluation suite complete"
echo "  BC-P + GAIA exit: $BC_GAIA_RC"
echo "  ALFWorld exit:    $ALFWORLD_RC"
echo "=============================================================="

if [ "$BC_GAIA_RC" -ne 0 ] || [ "$ALFWORLD_RC" -ne 0 ]; then
  exit 4
fi
