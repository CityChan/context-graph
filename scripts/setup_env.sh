#!/bin/bash
# FoldAgent environment setup for TACC Vista (agentfold conda env)
# Usage: conda activate agentfold && bash scripts/setup_env.sh
set -e

echo "=== Installing FoldAgent dependencies ==="

# Core ML
pip install transformers peft accelerate datasets

# Distributed / Training
pip install 'ray[default]' torchdata 'tensordict>=0.8.0,<=0.10.0,!=0.9.0'
pip install hydra-core omegaconf
pip install wandb tensorboard

# Web / API
pip install fastapi uvicorn httpx aiohttp pydantic openai

# Eval / Math
pip install math_verify latex2sympy2_extended pylatexenc

# Misc
pip install unidiff psutil pandas tqdm packaging codetiming dill 'numpy<2.0.0' 'pyarrow>=19.0.0'

# requirements.txt extras
pip install pybind11 pre-commit

echo ""
echo "=== Basic deps installed. ==="
echo ""
echo "NOTE: vllm/sglang not installed here (needs compute node for ARM build)."
echo "To install vllm on a compute node:"
echo "  idev -p gh -N 1 -n 1 -t 01:00:00"
echo "  conda activate agentfold"
echo "  pip install vllm"
