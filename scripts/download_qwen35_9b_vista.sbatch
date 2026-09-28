#!/bin/bash
#SBATCH --job-name=download-qwen35-9b
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=download-qwen35-9b-%j.log
set -eo pipefail
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate deepseek_v4
set -u
export HF_HOME="${SCRATCH:?Missing SCRATCH}/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
mkdir -p "$HF_HUB_CACHE"
python -u -c 'import json,os; from pathlib import Path; from huggingface_hub import snapshot_download; p=Path(snapshot_download("Qwen/Qwen3.5-9B",cache_dir=os.environ["HF_HUB_CACHE"])); assert p.resolve().is_relative_to(Path(os.environ["SCRATCH"]).resolve()); idx=json.loads((p/"model.safetensors.index.json").read_text()); shards=set(idx["weight_map"].values()); assert shards and all((p/s).is_file() and (p/s).stat().st_size>0 for s in shards); print("CHECKPOINT_READY",p,flush=True)'
