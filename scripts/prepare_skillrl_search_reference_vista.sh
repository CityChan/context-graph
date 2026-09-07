#!/bin/bash

# Prepare a pinned, independent SkillRL/Search-R1 reference stack on Vista.
# Invoke this through the one-node sbatch wrapper; do not run it inside the
# cxtgraph environment.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
REFERENCE_ROOT=${REFERENCE_ROOT:-${SCRATCH:?SCRATCH must be set}/skillrl_search_reference}
SKILLRL_ROOT=${SKILLRL_ROOT:-$REFERENCE_ROOT/SkillRL}
SKILLRL_COMMIT=${SKILLRL_COMMIT:-8e66726ed866a4e0a7f053586a41022798192e6c}
ENV_NAME=${ENV_NAME:-skillrl_search}
SOURCE_ENV_NAME=${SOURCE_ENV_NAME:-cxtgraph}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
TRAIN_PER_SOURCE=${TRAIN_PER_SOURCE:-128}
VAL_PER_SOURCE=${VAL_PER_SOURCE:-64}
DOWNLOAD_RL=${DOWNLOAD_RL:-0}

MODEL_ROOT=$REFERENCE_ROOT/models
DATA_ROOT=$REFERENCE_ROOT/data
RETRIEVER_ROOT=$REFERENCE_ROOT/retriever
QWEN_MODEL=$MODEL_ROOT/Qwen2.5-7B-Instruct
SFT_MODEL=$MODEL_ROOT/Search-7B-SFT
E5_MODEL=$MODEL_ROOT/e5-base-v2
RL_CHECKPOINT=$MODEL_ROOT/Search-7B-RL-checkpoint
RL_MODEL=$MODEL_ROOT/Search-7B-RL-HF

mkdir -p "$REFERENCE_ROOT" "$MODEL_ROOT" "$DATA_ROOT" "$RETRIEVER_ROOT" "$HF_HOME"

if [ ! -d "$SKILLRL_ROOT/.git" ]; then
  git clone https://github.com/aiming-lab/SkillRL.git "$SKILLRL_ROOT"
fi
git -C "$SKILLRL_ROOT" fetch origin "$SKILLRL_COMMIT"
git -C "$SKILLRL_ROOT" checkout --detach "$SKILLRL_COMMIT"
test "$(git -C "$SKILLRL_ROOT" rev-parse HEAD)" = "$SKILLRL_COMMIT"

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" --clone "$SOURCE_ENV_NAME"
fi
conda activate "$ENV_NAME"

if ! python -c "import faiss; assert hasattr(faiss, 'GpuMultipleClonerOptions')" >/dev/null 2>&1; then
  # A cloned Vista environment retains exact defaults-channel Python/libuuid
  # specs in its history. The ARM CUDA Faiss build is from conda-forge and
  # requires the matching conda-forge Python ABI and modern libuuid. Let conda
  # migrate those base packages inside this independent environment only.
  conda install -y -n "$ENV_NAME" --override-channels -c conda-forge --update-specs "python=3.10.19" "libuuid>=2.41.3" "numpy<2" "cuda-version=12.8" "faiss-gpu=1.9.0"
fi
python -m pip install --upgrade "huggingface_hub[cli]"
python -m pip install -e "$SKILLRL_ROOT"
python -m pip install -e "$SKILLRL_ROOT/agent_system/environments/env_package/search/third_party"
python -m pip install "gym==0.26.2"

python -c "import faiss, gym, ray, torch, transformers, vllm; assert hasattr(faiss, 'GpuMultipleClonerOptions'); print({'torch': torch.__version__, 'transformers': transformers.__version__, 'ray': ray.__version__, 'vllm': vllm.__version__, 'gym': gym.__version__, 'faiss': faiss.__version__})"

export HF_HOME
hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$QWEN_MODEL"
hf download Jianwen/Search-7B-SFT --local-dir "$SFT_MODEL"
hf download intfloat/e5-base-v2 --local-dir "$E5_MODEL"

INDEX_FILE=$RETRIEVER_ROOT/e5_Flat.index
CORPUS_FILE=$RETRIEVER_ROOT/wiki-18.jsonl
if [ ! -s "$INDEX_FILE" ]; then
  hf download PeterJinGo/wiki-18-e5-index --repo-type dataset --include part_aa part_ab --local-dir "$RETRIEVER_ROOT"
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
  mv "$CORPUS_FILE.partial" "$CORPUS_FILE"
  rm -f "$RETRIEVER_ROOT/wiki-18.jsonl.gz"
fi

cd "$SKILLRL_ROOT"
python examples/data_preprocess/preprocess_search_r1_dataset.py --hf_repo_id PeterJinGo/nq_hotpotqa_train --local_dir "$DATA_ROOT/full"

python "$PROJECT_ROOT/scripts/sample_searchr1_reference_data.py" --input "$DATA_ROOT/full/train.parquet" --output "$DATA_ROOT/train_diag.parquet" --per-source "$TRAIN_PER_SOURCE" --sources searchR1_nq,searchR1_hotpotqa
python "$PROJECT_ROOT/scripts/sample_searchr1_reference_data.py" --input "$DATA_ROOT/full/test.parquet" --output "$DATA_ROOT/test_diag.parquet" --per-source "$VAL_PER_SOURCE"

if [ "$DOWNLOAD_RL" = 1 ]; then
  hf download Jianwen/Search-7B-RL --include "actor/model_world_size_4_rank_*.pt" "actor/*.json" "actor/*.jinja" "actor/*.txt" --local-dir "$RL_CHECKPOINT"
  if [ ! -s "$RL_MODEL/config.json" ]; then
    python scripts/model_merger.py merge --backend fsdp --local_dir "$RL_CHECKPOINT/actor" --target_dir "$RL_MODEL"
  fi
fi

python -c "from pathlib import Path; required=[Path('$QWEN_MODEL/config.json'),Path('$SFT_MODEL/config.json'),Path('$E5_MODEL/config.json'),Path('$INDEX_FILE'),Path('$CORPUS_FILE'),Path('$DATA_ROOT/train_diag.parquet'),Path('$DATA_ROOT/test_diag.parquet')]; missing=[str(p) for p in required if not p.is_file() or p.stat().st_size == 0]; assert not missing, f'missing artifacts: {missing}'; print('reference artifacts verified')"

echo "Prepared SkillRL reference stack at $REFERENCE_ROOT"
echo "Pinned SkillRL commit: $SKILLRL_COMMIT"
echo "Next: start a fresh two-node idev and run scripts/run_skillrl_search_reference_2node_idev.sh"
