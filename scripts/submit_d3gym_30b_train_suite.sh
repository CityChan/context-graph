#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"

for method in react fold ctxgraph; do
  job_id=$(sbatch --parsable --export=ALL,D3GYM_MODE=train,D3GYM_METHOD="$method" scripts/run_d3gym_30b_instruct_8node.sh)
  echo "submitted D3-Gym 30B $method training: $job_id"
done
