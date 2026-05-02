#!/bin/bash
#SBATCH -J build-hp-idx
#SBATCH -o build-hp-idx.%j.out
#SBATCH -e build-hp-idx.%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 01:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# Build the HotpotQA Qwen3-Embedding index, one-shot. Outputs:
#   data/hotpotqa_corpus_embeddings.pkl
# Once this finishes, the search server can serve over the local pkl
# instead of the BrowseComp HF defaults.
#
# Pre-flight (login node):
#   python scripts/build_hotpotqa_corpus.py    # writes data/hotpotqa_corpus.parquet
# Then sbatch this script. The default embedder (Qwen3-Embedding-4B) was
# picked so that the trained search server can co-locate on a GPU shared
# with a 30B trainer; override via EMBED_MODEL env var if you have a
# dedicated search node.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Conda + CUDA ──
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}
BATCH_SIZE=${BATCH_SIZE:-32}

CORPUS=data/hotpotqa_corpus.parquet
OUT=data/hotpotqa_corpus_embeddings.pkl

if [ ! -f "$CORPUS" ]; then
  echo "ERROR: $CORPUS not found. Build it first on a login node:"
  echo "  python scripts/build_hotpotqa_corpus.py"
  exit 1
fi

echo "=============================================================="
echo "  HotpotQA index build"
echo "  Embedder: $EMBED_MODEL"
echo "  Corpus:   $CORPUS"
echo "  Output:   $OUT"
echo "  Started:  $(date)"
echo "=============================================================="

# Cache the embedder snapshot on this node if it's missing (the encoder
# loads via from_pretrained and will hit HF if the cache is empty).
SNAPSHOT_DIR="$HF_HOME/hub/models--$(echo "$EMBED_MODEL" | sed 's|/|--|g')/snapshots"
if [ ! -d "$SNAPSHOT_DIR" ] || [ -z "$(ls -A "$SNAPSHOT_DIR" 2>/dev/null)" ]; then
  echo "--- Embedder not cached, downloading on this node ---"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    huggingface-cli download "$EMBED_MODEL" --cache-dir "$HF_HOME" || {
      echo "Embedder download failed."
      exit 1
    }
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/build_hotpotqa_index.py \
  --corpus "$CORPUS" \
  --out "$OUT" \
  --model "$EMBED_MODEL" \
  --batch_size "$BATCH_SIZE"

echo "=============================================================="
echo "  Index build done at $(date)"
echo "  ls -lh $OUT"
ls -lh "$OUT"
echo "=============================================================="
