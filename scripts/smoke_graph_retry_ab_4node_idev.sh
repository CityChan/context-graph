#!/bin/bash
# Inference-only paired experiment; one shared server startup, separate artifacts.
set -euo pipefail
export PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
: "${SLURM_JOB_ID:?Run inside an existing four-node idev}"
: "${SCRATCH:?Missing SCRATCH}"
[[ $(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l) -eq 4 ]] || { echo 'Requires exactly four nodes'; exit 2; }
git diff --quiet HEAD -- agents scripts || { echo 'Commit experiment code before running'; exit 2; }
export MODEL_ID=${MODEL_ID:-Qwen/Qwen3.8-27B}
# Preserve an explicitly supplied offline checkpoint; resolver validates it.
export MODEL_PATH=${MODEL_PATH:-}
export BENCHMARK=bcp MEMORY_MODE=repaired
export GRAPH_ACTION_PROTOCOL=${GRAPH_ACTION_PROTOCOL:-controller}
export EVAL_CONTEXT_LENGTH=65536 MODEL_MAX_LEN=65536
export AGENT_BUDGET_MODE=working_context
export GRAPH_RETRY_AB=1 GRAPH_RETRY_GUARD=0
export GRAPH_RETRY_FAILURE_LIMIT=${GRAPH_RETRY_FAILURE_LIMIT:-3}
export SAMPLES=${SAMPLES:-3} WORKERS=${WORKERS:-1} SEED=${SEED:-42}
# Worktrees can share the immutable evaluation dataset, not logs or results.
export DATA_PATH=${DATA_PATH:-/work/09281/chc_1996/vista/context-graph/data/bc_test.parquet}
SUITE_PARENT=$(mktemp -d "$SCRATCH/graph-retry-ab-${SLURM_JOB_ID}-XXXXXX")
export RUN_ROOT="$SUITE_PARENT/run"
printf 'GRAPH_RETRY_AB model=%s context=65536 samples_per_arm=%s protocol=%s artifacts=%s\n' "$MODEL_ID" "$SAMPLES" "$GRAPH_ACTION_PROTOCOL" "$RUN_ROOT"
bash scripts/eval_bcp_qwen38_4node_idev.sh contextgraph
set +u
source "${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}"
conda activate "${AGENT_CONDA_ENV:-cxtgraph}"
set -u
python scripts/audit_graph_retry_ab.py "$RUN_ROOT"
