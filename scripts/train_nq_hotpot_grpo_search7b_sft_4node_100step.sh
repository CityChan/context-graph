#!/bin/bash
#SBATCH -J search7b-grpo100
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/search7b-grpo100.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/search7b-grpo100.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# One hundred-step vanilla GRPO run from the official SkillRL Search SFT
# checkpoint and static Search SkillBank. Node 0 serves Wiki-18 search;
# nodes 1-3 train the policy.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}

cd "$PROJECT_ROOT"
mkdir -p logs

if [ -n "${WORK:-}" ] && [ -f "$WORK/.wandb_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.wandb_env"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY is not set; expected it in the environment or ${WORK:-<WORK>}/.wandb_env"
  exit 1
fi

export MODEL_PATH=Jianwen/Search-7B-SFT
export ADV_ESTIMATOR=grpo
export TRAINER_VAL_ONLY=False
export VAL_BEFORE_TRAIN=True
export TOTAL_TRAINING_STEPS=100
export TEST_FREQ=10
export SAVE_FREQ=25
export VAL_MAX_SAMPLES=256
export TRAIN_BATCH_SIZE=48
export PPO_MINI_BATCH_SIZE=48
export ROLLOUT_N=4
export TRAIN_LR=1e-6
export LR_WARMUP_STEPS_RATIO=0.1
export ACTOR_KL_LOSS_COEF=0.001
export PROMPT_LENGTH=4096
export RESPONSE_LENGTH=2048
export CONTEXT_LENGTH=6144
export MAX_TURN=4
export MAX_SESSION=1
export VAL_MAX_SESSION=1
export TURN_MAX_NEW_TOKENS=384
export FINAL_ANSWER_RESERVE=512
export MUST_SEARCH=False
export USE_SKILLS_ONLY_MEMORY=True
export SKILLS_JSON_PATH="$PROJECT_ROOT/memory_data/search/claude_style_skills_search.json"
export SKILLS_TOP_K=6
export RUN_TAG=search7b_sft_nq_hotpot_grpo100_v2_fixed

exec bash scripts/train_nq_hotpot_grpo_qwen25_7b_4node_idev.sh
