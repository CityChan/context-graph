#!/bin/bash
#SBATCH -J bcp-memory-ablation
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 06:00:00
#SBATCH -A AST24021
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  sbatch --export=ALL "$0"
  exit 0
fi
# Base Qwen3-8B, greedy decoding, 64K, shared finalizer, no StructMem.
RUN_ROOT=${RUN_ROOT:-$PROJECT_ROOT/outputs/memory-ablation-${SLURM_JOB_ID}-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/suite.log") 2>&1
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
python -c 'import ray, torch, pyarrow'
if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Installing missing preflight dependency pytest into the active cxtgraph environment"
  python -m pip install --disable-pip-version-check --retries 2 --timeout 30 'pytest>=7,<9'
fi
python -m pytest -q tests/test_context_graph_foldagent_replay.py tests/test_context_graph_memory.py tests/test_context_graph_modes.py tests/test_graph_controller.py
python scripts/prepare_contextgraph_diagnostic.py --source data/bc_test.parquet --output "$RUN_ROOT/data" --samples "${SAMPLES:-24}" --seed "${SEED:-42}"
export MODEL_PATH=Qwen/Qwen3-8B
export BC_PROMPT_LENGTH=8192 BC_RESPONSE_LENGTH=57344 BC_MAX_TOKEN_LEN_PER_GPU=65536
export BC_CONTEXT_LENGTH=65536
export BC_MAX_TURN=100 BC_MAX_SESSION=10 BC_FINAL_ANSWER_RESERVE=1024
export BC_SESSION_TIMEOUT=3600 BC_TURN_MAX_NEW_TOKENS=2048
export BC_CONSOLIDATION_INTERVAL=5 BC_AUTO_PRUNE_MAX_ACTIVE=12
export STRUCTURED_MEMORY_ENABLED=0 STRUCTURED_MEMORY_REQUIRED=0 BC_DISABLE_WANDB=1
export BC_CAPTURE_MODEL_CONTEXTS=True BC_VAL_MAX_SAMPLES=-1
for variant in foldagent equivalent legacy repaired; do
  export BC_METHOD=contextgraph BC_CTXGRAPH_PROTOCOL=controller
  export BC_CONTROLLER_ACTION_POLICY=structural
  export BC_VAL_PARQUET="$RUN_ROOT/data/graph.parquet"
  export BC_MEMORY_MODE="$variant"
  if [ "$variant" = foldagent ]; then
    export BC_METHOD=foldagent BC_CTXGRAPH_PROTOCOL=legacy BC_MEMORY_MODE=legacy
    export BC_VAL_PARQUET="$RUN_ROOT/data/foldagent.parquet"
  elif [ "$variant" = equivalent ]; then
    export BC_MEMORY_MODE=foldagent
  fi
  export EXPERIMENT_NAME="memory_ablation_${SLURM_JOB_ID}_${variant}"
  export BC_VALIDATION_DATA_DIR="$RUN_ROOT/$variant"
  bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh
done
python scripts/audit_contextgraph_ablation.py "$RUN_ROOT"
