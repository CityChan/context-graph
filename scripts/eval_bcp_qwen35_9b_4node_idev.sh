#!/bin/bash
# Share the tested 27B serving stack and evaluation protocol.
set -euo pipefail
export MODEL_ID=Qwen/Qwen3.5-9B
# Resolve this model's own scratch snapshot instead of inheriting a 27B path.
unset MODEL_PATH
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_bcp_qwen38_4node_idev.sh" "$@"
