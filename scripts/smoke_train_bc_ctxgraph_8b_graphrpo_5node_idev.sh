#!/bin/bash
#SBATCH -J graphrpo-8b-smoke
#SBATCH -o logs/graphrpo-8b-smoke.%j.out
#SBATCH -e logs/graphrpo-8b-smoke.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 02:00:00
#SBATCH -A AST24021

# One-step end-to-end frozen-reference answer-likelihood GraphRPO smoke on a
# 4/5-node Vista allocation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SFT_FSDP_CHECKPOINT=${SFT_FSDP_CHECKPOINT:-${SCRATCH:-/scratch/09281/chc_1996}/contextgraph_sft_checkpoints/miroverse_full_policy_qwen3_8b_bs16_1ep_v1/global_step_174}
MODEL_PATH=${MODEL_PATH:-${SCRATCH:-/scratch/09281/chc_1996}/contextgraph_sft_models/miroverse_full_policy_qwen3_8b_bs16_1ep_v1_step174_hf}

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ ! -s "$MODEL_PATH/config.json" ] || ! find -L "$MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  if [ ! -d "$SFT_FSDP_CHECKPOINT" ]; then
    echo "ERROR: SFT FSDP checkpoint is missing: $SFT_FSDP_CHECKPOINT"
    exit 1
  fi
  if [ -e "$MODEL_PATH" ] && [ -n "$(find "$MODEL_PATH" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "ERROR: incomplete merged-model directory is not empty: $MODEL_PATH"
    exit 1
  fi
  mkdir -p "$(dirname "$MODEL_PATH")"
  echo "Merging SFT checkpoint: $SFT_FSDP_CHECKPOINT -> $MODEL_PATH"
  python -m verl.model_merger merge --backend fsdp --local_dir "$SFT_FSDP_CHECKPOINT" --target_dir "$MODEL_PATH"
fi
export MODEL_PATH

mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-${#NODELIST[@]}}
if [ "${#NODELIST[@]}" -ne "$EXPECTED_NUM_NODES" ] || [ "$EXPECTED_NUM_NODES" -lt 4 ]; then
  echo "ERROR: GraphRPO smoke requires at least four allocated nodes; got ${#NODELIST[@]}"
  exit 1
fi
export EXPECTED_NUM_NODES
TRAINER_NODES=$((EXPECTED_NUM_NODES - 1))
mkdir -p "$PROJECT_ROOT/logs"
export GRAPH_RPO_CREDIT_BACKEND=${GRAPH_RPO_CREDIT_BACKEND:-reference_answer_likelihood}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$TRAINER_NODES}
export ROLLOUT_N=${ROLLOUT_N:-2}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-$TRAIN_BATCH_SIZE}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-$TRAIN_BATCH_SIZE}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TEST_FREQ=${TEST_FREQ:-0}
export SAVE_FREQ=${SAVE_FREQ:-1}
export MAX_SESSION=${MAX_SESSION:-3}
export VAL_MAX_SESSION=${VAL_MAX_SESSION:-3}
export MAX_TURN=${MAX_TURN:-40}
export RUN_TAG=${RUN_TAG:-graphrpo_sft_1step_smoke}
export BC_DISABLE_WANDB=${BC_DISABLE_WANDB:-0}
export SAVE_ROLLOUT_DATA=${SAVE_ROLLOUT_DATA:-1}

echo "GraphRPO credit backend: $GRAPH_RPO_CREDIT_BACKEND"
bash scripts/train_bc_ctxgraph_8b_graphrpo_5node_48h.sh
