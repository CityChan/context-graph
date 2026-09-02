#!/bin/bash
#SBATCH -J miro-full-policy
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/miro-full-policy.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/miro-full-policy.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 08:00:00
#SBATCH -A AST24021

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
: "${START_INDEX:?START_INDEX is required}"
: "${MAX_SAMPLES:?MAX_SAMPLES is required}"

export EXPECTED_NUM_NODES=4
export TEACHER_TP=4
export COLOCATE_SEARCH=1
export DATA_PATH=${DATA_PATH:-$SCRATCH/contextgraph_sft/miroverse_full_policy/seeds.parquet}
export LOCAL_SEARCH_CORPUS=${LOCAL_SEARCH_CORPUS:-$SCRATCH/contextgraph_sft/miroverse_full_policy/retrieval_corpus.parquet}
export LOCAL_SEARCH_EMBEDDINGS=${LOCAL_SEARCH_EMBEDDINGS:-$SCRATCH/contextgraph_sft/miroverse_full_policy/retrieval_embeddings.pkl}
export NUM_WORKERS=${NUM_WORKERS:-2}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.70}
export FULL_POLICY_CURATOR=1
export RUN_TAG=${RUN_TAG:-${SLURM_JOB_ID}_miroverse_full_policy_s${START_INDEX}_n${MAX_SAMPLES}}
export ARTIFACT_ROOT=${ARTIFACT_ROOT:-$SCRATCH/contextgraph_sft/miroverse_full_policy_native/shards/$RUN_TAG}

test -s "$DATA_PATH"
test -s "$LOCAL_SEARCH_CORPUS"
test -s "$LOCAL_SEARCH_EMBEDDINGS"
mkdir -p "$ARTIFACT_ROOT"
printf 'start_index=%s\nmax_samples=%s\nrun_tag=%s\n' "$START_INDEX" "$MAX_SAMPLES" "$RUN_TAG" > "$ARTIFACT_ROOT/shard_request.txt"

cd "$PROJECT_ROOT"
bash scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh
touch "$ARTIFACT_ROOT/.complete"
