#!/bin/bash
#SBATCH -J miro-remain-prep
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/miro-remain-prep.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/miro-remain-prep.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -A AST24021

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
MIROVERSE_ROOT=${MIROVERSE_ROOT:-$SCRATCH/datasets/MiroVerse-v0.1/jsonl_sft}
OUTPUT_DIR=${OUTPUT_DIR:-$SCRATCH/contextgraph_sft/miroverse_remaining_full_policy}

SUBSETS=(
  MiroVerse-2WikiMultihopQA
  MiroVerse-HotpotQA
  MiroVerse-MegaScience
  MiroVerse-OneGen-TrainDataset-MultiHopQA
  MiroVerse-QA-Expert-Multi-Hop-V1.0
  MiroVerse-TaskCraft
  MiroVerse-Voyager1.0
  MiroVerse-WebDancer
  MiroVerse-WebShaper
  MiroVerse-WebWalkerQA-Silver
  MiroVerse-WikiTables
)
INPUTS=()
for subset in "${SUBSETS[@]}"; do
  input="$MIROVERSE_ROOT/$subset.jsonl"
  test -s "$input" || { echo "ERROR: missing MiroVerse subset: $input"; exit 2; }
  INPUTS+=("$input")
done

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate deepseek_v4
cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
python -u scripts/prepare_miroverse_retrieval_corpus.py --input "${INPUTS[@]}" --output "$OUTPUT_DIR/retrieval_corpus.parquet" --manifest "$OUTPUT_DIR/retrieval_corpus_manifest.json" --max-samples 0
HF_HOME=/work/09281/chc_1996/vista/cache HF_HUB_CACHE=/work/09281/chc_1996/vista/cache/hub python -u scripts/embed_local_search_corpus.py --input "$OUTPUT_DIR/retrieval_corpus.parquet" --output "$OUTPUT_DIR/retrieval_embeddings.pkl" --manifest "$OUTPUT_DIR/retrieval_embeddings_manifest.json" --model Qwen/Qwen3-Embedding-8B --batch-size 16 --max-length 2048 --attn-implementation sdpa
python -u scripts/prepare_miroverse_search_policy_seeds.py --input "${INPUTS[@]}" --output "$OUTPUT_DIR/seeds.parquet" --manifest "$OUTPUT_DIR/seeds_manifest.json" --max-samples 0
touch "$OUTPUT_DIR/.complete"
