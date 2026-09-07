#!/bin/bash

# Run on a Vista login node. This prepares the complete independent
# NQ/HotpotQA + Wiki-18 stack and does not clone or install SkillRL/Search-R1.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
DATA_ROOT=${DATA_ROOT:-${SCRATCH:?SCRATCH must be set}/context-graph-data/searchr1_nq_hotpotqa}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}
DOWNLOAD_MODEL=${DOWNLOAD_MODEL:-1}
VALIDATION_PER_SOURCE=${VALIDATION_PER_SOURCE:-128}
RETRIEVER_ROOT=${RETRIEVER_ROOT:-$DATA_ROOT/wiki18}
RETRIEVER_ENV=${RETRIEVER_ENV:-$DATA_ROOT/wiki18_retriever_env}
E5_MODEL_DIR=${E5_MODEL_DIR:-$RETRIEVER_ROOT/e5-base-v2}
PREPARE_RETRIEVER_ENV=${PREPARE_RETRIEVER_ENV:-1}
PREBUILD_CORPUS_CACHE=${PREBUILD_CORPUS_CACHE:-1}

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
export HF_HOME HF_HUB_CACHE HF_HUB_DISABLE_FILE_LOCKING=1
cd "$PROJECT_ROOT"

RAW_DIR=$DATA_ROOT/raw
PROCESSED_DIR=$DATA_ROOT/processed
mkdir -p "$RAW_DIR" "$PROCESSED_DIR" "$RETRIEVER_ROOT"

echo "Downloading NQ/HotpotQA parquet files into $RAW_DIR"
hf download PeterJinGo/nq_hotpotqa_train --repo-type dataset --include train.parquet test.parquet --local-dir "$RAW_DIR"
python scripts/prepare_nq_hotpot_search_data.py --train-input "$RAW_DIR/train.parquet" --validation-input "$RAW_DIR/test.parquet" --output-dir "$PROCESSED_DIR" --validation-per-source "$VALIDATION_PER_SOURCE"

if [ "$DOWNLOAD_MODEL" = 1 ]; then
  echo "Ensuring $MODEL_PATH is present in $HF_HUB_CACHE"
  hf download "$MODEL_PATH"
fi

echo "Downloading the official Search-R1 Wiki-18 corpus, E5 index, and encoder"
INDEX_FILE=$RETRIEVER_ROOT/e5_Flat.index
CORPUS_FILE=$RETRIEVER_ROOT/wiki-18.jsonl
if [ ! -s "$INDEX_FILE" ]; then
  hf download PeterJinGo/wiki-18-e5-index --repo-type dataset --include part_aa part_ab --local-dir "$RETRIEVER_ROOT"
  test -s "$RETRIEVER_ROOT/part_aa"
  test -s "$RETRIEVER_ROOT/part_ab"
  EXPECTED_INDEX_SIZE=$(($(stat -c %s "$RETRIEVER_ROOT/part_aa") + $(stat -c %s "$RETRIEVER_ROOT/part_ab")))
  cat "$RETRIEVER_ROOT/part_aa" "$RETRIEVER_ROOT/part_ab" > "$INDEX_FILE.partial"
  test "$(stat -c %s "$INDEX_FILE.partial")" -eq "$EXPECTED_INDEX_SIZE"
  mv "$INDEX_FILE.partial" "$INDEX_FILE"
  rm -f "$RETRIEVER_ROOT/part_aa" "$RETRIEVER_ROOT/part_ab"
fi
if [ ! -s "$CORPUS_FILE" ]; then
  hf download PeterJinGo/wiki-18-corpus --repo-type dataset --include wiki-18.jsonl.gz --local-dir "$RETRIEVER_ROOT"
  gzip -t "$RETRIEVER_ROOT/wiki-18.jsonl.gz"
  gzip -cd "$RETRIEVER_ROOT/wiki-18.jsonl.gz" > "$CORPUS_FILE.partial"
  test -s "$CORPUS_FILE.partial"
  mv "$CORPUS_FILE.partial" "$CORPUS_FILE"
  rm -f "$RETRIEVER_ROOT/wiki-18.jsonl.gz"
fi
hf download intfloat/e5-base-v2 --local-dir "$E5_MODEL_DIR"

if [ "$PREPARE_RETRIEVER_ENV" = 1 ]; then
  if [ ! -x "$RETRIEVER_ENV/bin/python" ]; then
    echo "Creating isolated Wiki-18 retriever environment at $RETRIEVER_ENV"
    conda create -y -p "$RETRIEVER_ENV" --override-channels -c conda-forge \
      "python=3.10.19" "libuuid>=2.41.3" "numpy<2" "cuda-version=12.8" \
      "faiss-gpu=1.9.0" pytorch transformers datasets fastapi uvicorn
  fi
  conda run -p "$RETRIEVER_ENV" python -c "import datasets, faiss, fastapi, torch, transformers, uvicorn; assert hasattr(faiss, 'GpuMultipleClonerOptions'); print({'faiss': faiss.__version__, 'torch': torch.__version__, 'transformers': transformers.__version__})"
fi

if [ "$PREBUILD_CORPUS_CACHE" = 1 ]; then
  echo "Materializing the Wiki-18 Arrow cache once on the login node"
  HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" conda run -p "$RETRIEVER_ENV" python -c "from datasets import load_dataset; corpus=load_dataset('json',data_files='$CORPUS_FILE',split='train',num_proc=8); print({'wiki18_rows':len(corpus),'columns':corpus.column_names})"
fi

python -c "from pathlib import Path; import json, pandas as pd; root=Path('$PROCESSED_DIR'); required=[root/'train.parquet',root/'validation.parquet',root/'validation_diag.parquet',root/'manifest.json',Path('$INDEX_FILE'),Path('$CORPUS_FILE'),Path('$E5_MODEL_DIR/config.json'),Path('$RETRIEVER_ENV/bin/python')]; missing=[str(p) for p in required if not p.is_file() or p.stat().st_size==0]; assert not missing, f'missing artifacts: {missing}'; frames={p.stem:pd.read_parquet(p) for p in required[:3]}; assert all(set(f.data_source)=={'searchR1_nq','searchR1_hotpotqa'} for f in frames.values()); print(json.dumps({k:{'rows':len(v),'sources':v.data_source.value_counts().to_dict()} for k,v in frames.items()},indent=2))"
echo "Prepared NQ/HotpotQA data at $PROCESSED_DIR"
echo "Prepared Wiki-18 retrieval stack at $RETRIEVER_ROOT"
