#!/bin/bash
#SBATCH -J miro-cg-full-rl8b-smoke
#SBATCH -o logs/miro-cg-full-rl8b-smoke.%j.out
#SBATCH -e logs/miro-cg-full-rl8b-smoke.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
MIRO_ROOT=${MIRO_ROOT:-$SCRATCH/contextgraph_sft/miroverse_full_policy}
SEEDS=${SEEDS:-$MIRO_ROOT/seeds.parquet}
TRAIN_DATA_FILE=${TRAIN_DATA_FILE:-$MIRO_ROOT/rl_smoke_train.parquet}
VAL_DATA_FILE=${VAL_DATA_FILE:-$MIRO_ROOT/rl_smoke_validation.parquet}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-$MIRO_ROOT/rl_smoke_split_manifest.json}

cd "$PROJECT_ROOT"
mkdir -p logs "$MIRO_ROOT"
python -u scripts/split_miroverse_rl_smoke_seeds.py --input "$SEEDS" --train-output "$TRAIN_DATA_FILE" --validation-output "$VAL_DATA_FILE" --manifest "$SPLIT_MANIFEST" --train-samples "${TRAIN_SAMPLES:-32}" --validation-samples "${VALIDATION_SAMPLES:-20}" --seed "${SPLIT_SEED:-42}"

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
export DATASET_LABEL=MiroVerse-MuSiQue
export TRAIN_DATA_FILE
export VAL_DATA_FILE
export LOCAL_SEARCH_CORPUS=${LOCAL_SEARCH_CORPUS:-$MIRO_ROOT/retrieval_corpus.parquet}
export LOCAL_SEARCH_EMBEDDINGS=${LOCAL_SEARCH_EMBEDDINGS:-$MIRO_ROOT/retrieval_embeddings.pkl}
export BC_CTXGRAPH_PROTOCOL=full_policy
export BC_DISABLE_WANDB=${BC_DISABLE_WANDB:-1}
export RUN_TAG=${RUN_TAG:-miroverse_full_policy_rl8b_5step}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-smoke_miroverse_ctxgraph_full_policy_rl_qwen3_8b_5step_${SLURM_JOB_ID:-idev}}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH/context-graph-ckpts/$EXPERIMENT_NAME}
export PROMPT_LENGTH=8192
export RESPONSE_LENGTH=24576
export CONTEXT_LENGTH=32768
export TRAIN_BATCH_SIZE=4
export ROLLOUT_N=2
export PPO_MINI_BATCH_SIZE=2
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-5}
export VAL_BEFORE_TRAIN=True
export TEST_FREQ=1
export SAVE_FREQ=${SAVE_FREQ:-1}
export TRAIN_LR=1e-6
export USE_KL_LOSS=False
export ACTOR_KL_LOSS_COEF=0.0
export ALGORITHM_KL_COEF=0.0
export CLIP_RATIO_LOW=0.2
export CLIP_RATIO_HIGH=0.28
export MAX_TURN=${MAX_TURN:-60}
export MAX_SESSION=${MAX_SESSION:-8}
export VAL_MAX_SESSION=${VAL_MAX_SESSION:-8}
export TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-2048}
export FINAL_ANSWER_RESERVE=${FINAL_ANSWER_RESERVE:-1024}
export SESSION_TIMEOUT=${SESSION_TIMEOUT:-1800}

exec bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
