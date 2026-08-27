#!/bin/bash
# One-time Vista setup for AppWorld ContextGraph trajectory generation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SOURCE_CONDA_ENV=${SOURCE_CONDA_ENV:-deepseek_v4}
APPWORLD_CONDA_ENV=${APPWORLD_CONDA_ENV:-appworld_cxtgraph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
APPWORLD_ROOT=${APPWORLD_ROOT:-$SCRATCH/contextgraph_deps/appworld}

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "$APPWORLD_CONDA_ENV"; then
  conda create --yes --name "$APPWORLD_CONDA_ENV" --clone "$SOURCE_CONDA_ENV"
fi
conda activate "$APPWORLD_CONDA_ENV"
set -u

python -m pip install -r "$PROJECT_ROOT/requirements_appworld_sft.txt"
mkdir -p "$APPWORLD_ROOT"
export APPWORLD_ROOT
cd "$APPWORLD_ROOT"
appworld install
appworld download data
appworld verify tests --root "$APPWORLD_ROOT"
appworld verify tasks --root "$APPWORLD_ROOT"
python "$PROJECT_ROOT/scripts/make_appworld_data.py" --count-only --seed 42
echo "AppWorld SFT environment ready: env=$APPWORLD_CONDA_ENV root=$APPWORLD_ROOT"
