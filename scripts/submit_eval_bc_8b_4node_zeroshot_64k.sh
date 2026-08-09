#!/bin/bash
# Submit the three full 150-example evaluations after all idev smoke tests pass.

set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"
mkdir -p logs

for method in baseline foldagent contextgraph; do
  job_id=$(BC_METHOD="$method" \
    BC_CONTEXT_LENGTH=65536 \
    BC_PROMPT_LENGTH=8192 \
    BC_RESPONSE_LENGTH=57344 \
    BC_YARN_FACTOR=2.0 \
    BC_YARN_ORIGINAL_LENGTH=32768 \
    BC_VAL_MAX_SAMPLES=-1 \
    sbatch --parsable \
    --job-name="eval-bc-8b-${method}-64k" \
    --output="logs/eval-bc-8b-${method}-64k.%j.out" \
    --error="logs/eval-bc-8b-${method}-64k.%j.err" \
    --nodes=4 \
    --time=04:00:00 \
    scripts/eval_bc_baseline_8b_4node_zeroshot.sh)
  echo "$method: submitted job $job_id"
done
