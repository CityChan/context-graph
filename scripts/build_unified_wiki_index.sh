#!/bin/bash
#SBATCH -J build-wiki-idx
#SBATCH -o build-wiki-idx.%j.out
#SBATCH -e build-wiki-idx.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# Build the unified Wikipedia Qwen3-Embedding index (HotpotQA + 2WikiMQA
# distractor pools merged). One-shot. Outputs:
#   data/wiki_corpus_embeddings.pkl
#
# Reuses scripts/build_hotpotqa_index.py — the encoder is corpus-agnostic.
#
# Pre-flight (login node):
#   python scripts/build_unified_wiki_corpus.py
# Then sbatch this script. The merged HotpotQA + 2WikiMQA corpus is
# ~840K articles; expect 2-3h to encode at batch_size=32 with
# Qwen3-Embedding-4B. Walltime set to 4h for headroom.
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

CORPUS=data/wiki_corpus.parquet
OUT=data/wiki_corpus_embeddings.pkl

if [ ! -f "$CORPUS" ]; then
  echo "ERROR: $CORPUS not found. Build it first on a login node:"
  echo "  python scripts/build_unified_wiki_corpus.py"
  exit 1
fi

echo "=============================================================="
echo "  Unified Wiki index build (HotpotQA + 2WikiMQA)"
echo "  Embedder: $EMBED_MODEL"
echo "  Corpus:   $CORPUS"
echo "  Output:   $OUT"
echo "  Started:  $(date)"
echo "=============================================================="

# Cache the embedder snapshot on this node if it's missing
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
