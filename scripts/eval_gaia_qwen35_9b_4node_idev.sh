#!/bin/bash
# GAIA text-only internal comparison using the existing local search corpus.
set -euo pipefail
export BENCHMARK=gaia
export MODEL_ID=Qwen/Qwen3.5-9B
export SAMPLES=${SAMPLES:--1}
unset MODEL_PATH
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_bcp_qwen38_4node_idev.sh" "$@"
