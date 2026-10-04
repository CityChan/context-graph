#!/usr/bin/env bash
# BC-P RL only: one retriever + one frozen helper + two trainers, two updates.
set -euo pipefail
export GRAM_BENCHMARK=bcp
export GRAM_ACTION_DECODING=${GRAM_ACTION_DECODING:-xml_regex}
exec bash "$(dirname "${BASH_SOURCE[0]}")/smoke_gram_rl_qwen35_9b_4node.sbatch" "$@"
