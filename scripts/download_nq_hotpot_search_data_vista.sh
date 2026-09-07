#!/bin/bash

# Run on a Vista login node. This does not clone or install SkillRL.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
DATA_ROOT=${DATA_ROOT:-${SCRATCH:?SCRATCH must be set}/context-graph-data/searchr1_nq_hotpotqa}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
DOWNLOAD_MODEL=${DOWNLOAD_MODEL:-1}
VALIDATION_PER_SOURCE=${VALIDATION_PER_SOURCE:-128}

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
export HF_HOME HF_HUB_CACHE HF_HUB_DISABLE_FILE_LOCKING=1
cd "$PROJECT_ROOT"

RAW_DIR=$DATA_ROOT/raw
PROCESSED_DIR=$DATA_ROOT/processed
mkdir -p "$RAW_DIR" "$PROCESSED_DIR"

echo "Downloading NQ/HotpotQA parquet files into $RAW_DIR"
hf download PeterJinGo/nq_hotpotqa_train --repo-type dataset --include train.parquet test.parquet --local-dir "$RAW_DIR"
python scripts/prepare_nq_hotpot_search_data.py --train-input "$RAW_DIR/train.parquet" --validation-input "$RAW_DIR/test.parquet" --output-dir "$PROCESSED_DIR" --validation-per-source "$VALIDATION_PER_SOURCE"

if [ "$DOWNLOAD_MODEL" = 1 ]; then
  echo "Ensuring $MODEL_PATH is present in $HF_HUB_CACHE"
  hf download "$MODEL_PATH"
fi

python -c "from pathlib import Path; import json, pandas as pd; root=Path('$PROCESSED_DIR'); required=[root/'train.parquet',root/'validation.parquet',root/'validation_diag.parquet',root/'manifest.json']; missing=[str(p) for p in required if not p.is_file() or p.stat().st_size==0]; assert not missing, f'missing artifacts: {missing}'; frames={p.stem:pd.read_parquet(p) for p in required[:3]}; assert all(set(f.data_source)=={'searchR1_nq','searchR1_hotpotqa'} for f in frames.values()); print(json.dumps({k:{'rows':len(v),'sources':v.data_source.value_counts().to_dict()} for k,v in frames.items()},indent=2))"
echo "Prepared data at $PROCESSED_DIR"
