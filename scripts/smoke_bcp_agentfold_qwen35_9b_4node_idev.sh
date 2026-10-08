#!/bin/bash
# Exercise search -> model-written fold -> tool dispatch on a small real run.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
: "${SLURM_JOB_ID:?Run inside an existing four-node idev}"
: "${SCRATCH:?Missing Vista scratch directory}"
export SAMPLES=${SAMPLES:-3} WORKERS=${WORKERS:-1} BENCHMARK=bcp MEMORY_MODE=repaired
export RUN_ROOT=${RUN_ROOT:-$(mktemp -d "$SCRATCH/bcp-agentfold-smoke-${SLURM_JOB_ID}-XXXXXX")}
export CONDA_SH=${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}
export AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
set +u
source "$CONDA_SH"
conda activate "$AGENT_CONDA_ENV"
set -u
cd "$PROJECT_ROOT"
rc=0
bash scripts/eval_bcp_qwen35_9b_4node_idev.sh agentfold || rc=1
python scripts/audit_agentfold_smoke.py "$RUN_ROOT" || rc=1
echo "AGENTFOLD_SMOKE_COMPLETE status=$rc artifacts=$RUN_ROOT"
exit "$rc"
