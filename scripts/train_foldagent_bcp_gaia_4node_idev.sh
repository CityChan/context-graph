#!/bin/bash
# Run one FoldGRPO experiment in an existing allocation; never submit a job.
# Usage: bash scripts/train_foldagent_bcp_gaia_4node_idev.sh bcp|gaia EXPECTED_JOB_ID
set -euo pipefail

TASK=${1:?Usage: $0 bcp|gaia EXPECTED_JOB_ID}
EXPECTED_JOB=${2:?Provide the intended idev job ID}
case "$TASK" in bcp|gaia) ;; *) echo "Expected bcp or gaia" >&2; exit 2 ;; esac
if [ "${SLURM_JOB_ID:-}" != "$EXPECTED_JOB" ]; then
  echo "Wrong allocation: expected $EXPECTED_JOB, got ${SLURM_JOB_ID:-none}" >&2
  exit 2
fi
cd /work/09281/chc_1996/vista/context-graph
mapfile -t ALLOCATION_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#ALLOCATION_NODES[@]}" -ne 4 ]; then
  echo "Requires four nodes: one search and three trainer ranks" >&2
  exit 2
fi
mkdir -p logs
exec 9>"logs/foldagent-rl-${SLURM_JOB_ID}.lock"
flock -n 9 || { echo "A FoldAgent RL wrapper is already using this allocation" >&2; exit 2; }
STAMP=$(date +%Y%m%d_%H%M%S)
export RUN_TAG="${TASK}_4n_32k_${SLURM_JOB_ID}_${STAMP}"
export EXPERIMENT_NAME="train_foldagent_${RUN_TAG}"
export CHECKPOINT_ROOT="${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/$EXPERIMENT_NAME"
RUN_LOG="logs/${EXPERIMENT_NAME}.log"
exec > >(tee -a "$RUN_LOG") 2>&1
echo "Task=$TASK job=$SLURM_JOB_ID nodes=${ALLOCATION_NODES[*]}"
echo "Commit=$(git rev-parse HEAD) log=$RUN_LOG checkpoint=$CHECKPOINT_ROOT"
squeue -j "$SLURM_JOB_ID" -o '%.18i %.10L %.6D %N'
echo "The allocation time limit still applies; 50 steps are a target, not a runtime guarantee."

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
if [ -f "${WORK:-/work/09281/chc_1996/vista}/.openai_env" ]; then
  source "${WORK:-/work/09281/chc_1996/vista}/.openai_env"
fi
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = dummy ]; then
  echo "Missing judge credentials in the environment" >&2
  exit 2
fi
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
bash scripts/check_qwen3_observation_tokens.sh

if [ "$TASK" = bcp ]; then
  export TRAIN_DATA_FILE=data/bc_train.parquet
  export VAL_DATA_FILE=data/bc_test.parquet
else
  # Use existing splits as-is. Build missing splits in a unique directory,
  # never overwrite data that another allocation might be reading.
  if [ -s data/gaia_train_branch.parquet ] && [ -s data/gaia_holdout_branch.parquet ]; then
    DATA_DIR=data
  else
    DATA_DIR="data/$EXPERIMENT_NAME"
    python scripts/split_gaia_validation_for_training.py --input-dir data --output-dir "$DATA_DIR" --holdout-fraction 0.2 --seed 42
  fi
  export TRAIN_DATA_FILE="$DATA_DIR/gaia_train_branch.parquet"
  export VAL_DATA_FILE="$DATA_DIR/gaia_holdout_branch.parquet"
  echo "GAIA uses the existing local-search corpus, matching the historical launcher."
fi
python -c 'import os, pandas as pd; from scripts.split_gaia_validation_for_training import _identity; frames=[pd.read_parquet(os.environ[k]) for k in ("TRAIN_DATA_FILE","VAL_DATA_FILE")]; assert all(len(f)>0 for f in frames), "Empty split"; print("Data rows: train=%d val=%d" % tuple(map(len,frames))); ids=[set(_identity(r)[0] for _,r in f.iterrows()) for f in frames] if "gaia" in os.environ["TRAIN_DATA_FILE"] else None; assert ids is None or ids[0].isdisjoint(ids[1]), "GAIA train/holdout overlap"'
sha256sum "$TRAIN_DATA_FILE" "$VAL_DATA_FILE"

export EXPECTED_NUM_NODES=4 MODEL_PATH=Qwen/Qwen3-8B
export PROMPT_LENGTH=8192 RESPONSE_LENGTH=24576 CONTEXT_LENGTH=32768
export TRAIN_BATCH_SIZE=6 ROLLOUT_N=8 PPO_MINI_BATCH_SIZE=6
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
export TRAINER_VAL_ONLY=False VAL_BEFORE_TRAIN=True TEST_FREQ=10 SAVE_FREQ=5
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 DATALOADER_NUM_WORKERS=0
export TRAIN_LR=1e-6 USE_KL_LOSS=False ACTOR_KL_LOSS_COEF=0.0 ALGORITHM_KL_COEF=0.0
export CLIP_RATIO_LOW=0.2 CLIP_RATIO_HIGH=0.28
export LORA_RANK=0 ENTROPY_FROM_LOGITS_WITH_CHUNKING=True
export MAX_TURN=100 MAX_SESSION=10 VAL_MAX_SESSION=10 TURN_MAX_NEW_TOKENS=2048
export FINAL_ANSWER_RESERVE=1024 FINAL_ANSWER_SAFETY_MARGIN=64 SESSION_TIMEOUT=3600
export BC_SEARCH_TIMEOUT_SECONDS=600
# Do not accidentally resume an unrelated run inherited from the terminal.
unset RESUME_CHECKPOINT_PATH RESUME_CHECKPOINT_ROOT
echo "FoldGRPO full-parameter: steps=$TOTAL_TRAINING_STEPS batch=$TRAIN_BATCH_SIZE rollout_n=$ROLLOUT_N context=$CONTEXT_LENGTH"
set +e
bash scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh
RUN_RC=$?
set -e
echo "FOLDAGENT_RL_DONE task=$TASK job=$SLURM_JOB_ID exit=$RUN_RC log=$RUN_LOG"
if [ -f "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt" ]; then
  echo "Latest saved checkpoint iteration:"
  cat "$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
else
  echo "No checkpoint tracker found; no saved training step is confirmed."
fi
exit "$RUN_RC"
