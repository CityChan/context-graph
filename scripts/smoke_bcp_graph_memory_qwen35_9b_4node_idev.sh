#!/bin/bash
# Run one method, or both sequentially, inside an existing four-node idev.
set -euo pipefail
METHOD=${1:-both}
case "$METHOD" in memobrain|amem|both) ;; *) echo 'Usage: bash scripts/smoke_bcp_graph_memory_qwen35_9b_4node_idev.sh memobrain|amem|both'; exit 2 ;; esac
export PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
: "${SLURM_JOB_ID:?Run inside an existing four-node idev}"
: "${SCRATCH:?Missing Vista scratch directory}"
export SAMPLES=${SAMPLES:-3} WORKERS=${WORKERS:-1} BENCHMARK=bcp
export MEMORY_MODE=repaired
export CONDA_SH=${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}
export AGENT_CONDA_ENV=${AGENT_CONDA_ENV:-cxtgraph}
set +u
source "$CONDA_SH"
conda activate "$AGENT_CONDA_ENV"
set -u
cd "$PROJECT_ROOT"
mapfile -t SMOKE_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#SMOKE_NODES[@]} == 4 ]] || { echo 'Exactly four nodes required'; exit 2; }
SMOKE_ROOT=${SMOKE_ROOT:-$(mktemp -d "$SCRATCH/bcp-graph-memory-${SLURM_JOB_ID}-XXXXXX")}
mkdir -p "$SMOKE_ROOT"
# Missing dependencies go into a scratch venv; keep the serving/search envs intact.
if [[ "$METHOD" == amem || "$METHOD" == both ]]; then
  if [[ -z ${AMEM_AGENT_PYTHON:-} ]]; then
    python scripts/prepare_amem_environment.py --root "$SCRATCH/context-graph-agent-benchmarks/envs" --python-file "$SMOKE_ROOT/amem-python.txt"
    AMEM_AGENT_PYTHON=$(cat "$SMOKE_ROOT/amem-python.txt")
  fi
  export AMEM_AGENT_PYTHON
  printf '%s\n' "$AMEM_AGENT_PYTHON" > "$SMOKE_ROOT/amem-python.txt"
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache/hub" "$AMEM_AGENT_PYTHON" scripts/prepare_amem_embedding.py
fi
METHODS=("$METHOD")
[[ "$METHOD" != both ]] || METHODS=(memobrain amem)
rc=0
for method in "${METHODS[@]}"; do
  export RUN_ROOT="$SMOKE_ROOT/$method"
  if ! bash scripts/eval_bcp_qwen35_9b_4node_idev.sh "$method"; then
    echo "GRAPH_MEMORY_SMOKE_FAILED method=$method; inspect $RUN_ROOT"
    rc=1
  fi
  # Completed failed runs still have useful trajectories; never skip their audit.
  python scripts/audit_graph_memory_smoke.py "$RUN_ROOT" || rc=1
done
echo "GRAPH_MEMORY_SMOKE_COMPLETE status=$rc artifacts=$SMOKE_ROOT"
exit "$rc"
