#!/bin/bash
# Clone QeRL's working environment and add only the ContextGraph agent-loop dependencies.
set -euo pipefail

CONDA_BASE=${CONDA_BASE:-/work/09281/chc_1996/vista/miniconda3}
SOURCE_ENV=${SOURCE_ENV:-qerl}
TARGET_ENV=${TARGET_ENV:-graphtrl}
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
QERL_ROOT=${QERL_ROOT:-/work/09281/chc_1996/vista/QeRL}

source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$TARGET_ENV"; then
  conda create -n "$TARGET_ENV" --clone "$SOURCE_ENV" -y
fi
conda activate "$TARGET_ENV"
python -m pip install "hydra-core" "omegaconf" "ray[default]" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "codetiming" "aiohttp" "httpx" "pydantic" "unidiff"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$QERL_ROOT:${PYTHONPATH:-}"
python -c "from omegaconf import OmegaConf; import tensordict, torch, trl, trl_agent.train, vllm; config=OmegaConf.load('recipes/trl_agent/foldagent_gsm8k.yaml'); assert config.algorithm.adv_estimator == 'foldgrpo'; print('graphtrl ready:', torch.__version__, trl.__version__, vllm.__version__, tensordict.__version__)"
