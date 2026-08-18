#!/bin/bash
# Submit matched 32K GRPO training on the GAIA train split for three methods.

set -euo pipefail

export GAIA_CONTEXT_TAG=32k
export GAIA_PROMPT_LENGTH=8192
export GAIA_RESPONSE_LENGTH=24576
export GAIA_CONTEXT_LENGTH=32768

exec bash scripts/submit_train_gaia_grpo_8b_5node_50step_64k.sh
