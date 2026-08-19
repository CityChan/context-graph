#!/bin/bash
set -euo pipefail

# Download/cache the OpenThoughts cold-start SFT checkpoint when necessary,
# then submit matched BC-P and GAIA evaluations for all three agent methods.

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run this submitter on a Vista login node, not inside an allocation"
  exit 1
fi

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_PATH=${MODEL_PATH:-open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
HF_CLI=${HF_CLI:-/work/09281/chc_1996/vista/miniconda3/envs/cxtgraph/bin/hf}
PYTHON_BIN=${PYTHON_BIN:-/work/09281/chc_1996/vista/miniconda3/envs/cxtgraph/bin/python}
DOWNLOAD_MODEL=${DOWNLOAD_MODEL:-1}
RUN_BC=${RUN_BC:-1}
RUN_GAIA=${RUN_GAIA:-1}

cd "$PROJECT_ROOT"
mkdir -p logs "$HF_HUB_CACHE"

MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
if ! compgen -G "$MODEL_CACHE_DIR/snapshots/*/config.json" >/dev/null; then
  if [ "$DOWNLOAD_MODEL" != "1" ]; then
    echo "ERROR: model is not cached at $MODEL_CACHE_DIR and DOWNLOAD_MODEL=$DOWNLOAD_MODEL"
    exit 2
  fi
  if [ ! -x "$HF_CLI" ]; then
    echo "ERROR: Hugging Face CLI not found at $HF_CLI"
    exit 3
  fi
  echo "Caching $MODEL_PATH under $HF_HUB_CACHE"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" "$HF_CLI" download "$MODEL_PATH"
fi

echo "Model cache ready: $MODEL_CACHE_DIR"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python executable not found at $PYTHON_BIN"
  exit 4
fi

echo "Checking model compatibility before submitting jobs"
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" \
  "$PYTHON_BIN" scripts/check_hf_model_support.py "$MODEL_PATH"

if [ "$RUN_BC" = "1" ]; then
  MODEL_PATH="$MODEL_PATH" \
  BC_JOB_MODEL_TAG=openthinker-sft-8b \
  BC_EXPERIMENT_MODEL_TAG=openthinker_sft_8b \
  bash scripts/submit_eval_bc_8b_4node_zeroshot_64k.sh
fi

if [ "$RUN_GAIA" = "1" ]; then
  GAIA_MODEL_PATH="$MODEL_PATH" \
  GAIA_JOB_MODEL_TAG=openthinker-sft-8b \
  GAIA_EXPERIMENT_MODEL_TAG=openthinker_sft_8b \
  GAIA_EVAL_MODE=zeroshot \
  bash scripts/submit_gaia_benchmark_8b_5node.sh
fi
