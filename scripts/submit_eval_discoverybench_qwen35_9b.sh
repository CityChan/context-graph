#!/bin/bash
set -euo pipefail
cd "${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}"
mkdir -p logs
exec sbatch scripts/eval_discoverybench_qwen35_9b_4node.sbatch
