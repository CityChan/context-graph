#!/bin/bash
#SBATCH -J ctxgraph-stage2-rpo
#SBATCH -o logs/ctxgraph-stage2-rpo.%j.out
#SBATCH -e logs/ctxgraph-stage2-rpo.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH -A AST24021

# Formal stage-2 evaluator-free GraphRPO specialization. Start from the
# completed MiroVerse controller-SFT checkpoint, then use paired QA probes from
# the pre-update policy to score graph states before and after valid edits.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH:-/scratch/09281/chc_1996}
SFT_FSDP_CHECKPOINT=${SFT_FSDP_CHECKPOINT:-$SCRATCH_ROOT/contextgraph_sft_checkpoints/miroverse_full_policy_qwen3_8b_bs16_1ep_v1/global_step_174}
MODEL_PATH=${MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/miroverse_full_policy_qwen3_8b_bs16_1ep_v1_step174_hf}

cd "$PROJECT_ROOT"
if [ ! -s "$MODEL_PATH/config.json" ] || ! find -L "$MODEL_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit 2>/dev/null | grep -q .; then
  if [ ! -d "$SFT_FSDP_CHECKPOINT" ]; then
    echo "ERROR: stage-1 SFT checkpoint is missing: $SFT_FSDP_CHECKPOINT"
    exit 1
  fi
  if [ -e "$MODEL_PATH" ] && [ -n "$(find "$MODEL_PATH" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "ERROR: incomplete merged-model directory is not empty: $MODEL_PATH"
    exit 1
  fi
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  mkdir -p "$(dirname "$MODEL_PATH")"
  echo "Merging stage-1 SFT checkpoint: $SFT_FSDP_CHECKPOINT -> $MODEL_PATH"
  python -m verl.model_merger merge --backend fsdp --local_dir "$SFT_FSDP_CHECKPOINT" --target_dir "$MODEL_PATH"
fi

export MODEL_PATH
export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-5}
export ADV_ESTIMATOR=graphrpo
export POLICY_LOSS_MODE=graphrpo
export BC_CTXGRAPH_PROTOCOL=controller
export PROCESS_REWARD_SPEC='[scope]'
export USE_KL_LOSS=True
export GRAPH_RPO_CREDIT_BACKEND=${GRAPH_RPO_CREDIT_BACKEND:-old_policy_counterfactual_qa}
export GRAPH_RPO_COUNTERFACTUAL_SAMPLES=${GRAPH_RPO_COUNTERFACTUAL_SAMPLES:-2}
export GRAPH_RPO_COUNTERFACTUAL_ENABLE_THINKING=${GRAPH_RPO_COUNTERFACTUAL_ENABLE_THINKING:-False}
export GRAPH_RPO_ALPHA=${GRAPH_RPO_ALPHA:-0.1}
export GRAPH_RPO_DELTA_MAX=${GRAPH_RPO_DELTA_MAX:-0.25}
export DATA_SEED=${DATA_SEED:-42}
export SAVE_ROLLOUT_DATA=${SAVE_ROLLOUT_DATA:-1}
export BC_DISABLE_WANDB=${BC_DISABLE_WANDB:-0}
export BC_REQUIRE_WANDB=${BC_REQUIRE_WANDB:-1}
export RUN_TAG=${RUN_TAG:-stage2_graphrpo_counterfactual_sft_5n_48h}

exec bash scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
