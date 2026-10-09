#!/bin/bash
# Exercise the summary path on short ScienceWorld observations. Not a benchmark.
set -euo pipefail
export STATEFUL_MEMORY_SMOKE=1 SAMPLES=2
exec bash "$(dirname "${BASH_SOURCE[0]}")/smoke_stateful_memory_qwen35_9b_4node_idev.sh" scienceworld supo
