#!/bin/bash
set -euo pipefail

mkdir -p logs

for spec in \
  "foldagent:scripts/train_bc_foldagent_8b_5node_50step_32k_active.sh" \
  "contextgraph:scripts/train_bc_ctxgraph_8b_5node_50step_32k_active.sh"
do
  method=${spec%%:*}
  script=${spec#*:}
  job_id=$(sbatch --parsable "$script")
  printf '%s: submitted job %s\n' "$method" "$job_id"
done

