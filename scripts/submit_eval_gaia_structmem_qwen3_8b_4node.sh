#!/bin/bash
#SBATCH -J gaia-structmem8b
#SBATCH -o logs/gaia-structmem8b.%j.out
#SBATCH -e logs/gaia-structmem8b.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 02:00:00
#SBATCH -A AST24021

# Submit or run the paper-aligned StructMem GAIA validation on four Vista GH200
# nodes. Invoking with `bash` on a login node self-submits; invoking with `sbatch`
# or from an allocation runs the evaluation directly.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
EVAL_TIME=${EVAL_TIME:-02:00:00}
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
  echo "Submitted four-node GAIA StructMem evaluation: $job_id"
  squeue -j "$job_id" -o '%.18i %.9P %.32j %.2t %.10M %.10L %.6D %R' || true
  exit 0
fi

export RUN_BC=0
export RUN_GAIA=1
export EVAL_VARIANT=base_structmem_v3
export EVAL_MODEL_TAG=qwen3_8b_base
export EVAL_CTXGRAPH_PROTOCOL=controller
export STRUCTURED_MEMORY_ENABLED=1
export STRUCTURED_MEMORY_REQUIRED=1
export STRUCTURED_MEMORY_GAP_INTERVAL=8
export STRUCTURED_MEMORY_GAP_JITTER=1
export STRUCTURED_MEMORY_CONTROLLER_RETRIES=2
export STRUCTURED_MEMORY_STOP_ON_READY=1
export STRUCTURED_MEMORY_STEP_LIMIT=40
export STRUCTURED_MEMORY_RELATION_CANDIDATES=128
export GAIA_MAX_TURN=100
export GAIA_MAX_SESSION=10
export GAIA_FINAL_ANSWER_RESERVE=1024
export BC_DISABLE_WANDB=1
export WANDB_MODE=disabled

echo "Starting paper-aligned StructMem GAIA validation on job $SLURM_JOB_ID"
exec bash scripts/eval_bc_gaia_qwen3_8b_base_idev.sh
