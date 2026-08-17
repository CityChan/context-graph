#!/bin/bash
# Submit matched ReAct, FoldAgent, and ContextGraph Qwen3-30B SAB evaluations.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"

TS=$(date +%Y%m%d_%H%M%S)
for method in react fold ctxgraph; do
  SAB_METHOD="$method" EXPERIMENT_NAME="eval_${method}_sab_30b_instruct_8n_formal_${TS}" \
    bash scripts/submit_sab_react_30b_instruct_8node_formal.sh
done

echo "Submitted matched SAB formal suite: ReAct, FoldAgent, ContextGraph"
