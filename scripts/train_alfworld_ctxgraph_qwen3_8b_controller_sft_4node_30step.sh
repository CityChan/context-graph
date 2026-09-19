#!/bin/bash
#SBATCH -J cg-alf-sft-rl
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/cg-alf-sft-rl.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/cg-alf-sft-rl.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A AST24021

# ContextGraph FoldGRPO initialized from the merged MiroVerse controller-SFT
# checkpoint. This trains the whole policy on ALFWorld; it is not another SFT
# pass and it does not freeze the environment-action portion of the model.
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
export PROJECT_ROOT
export MODEL_PATH=${MODEL_PATH:-$SCRATCH/contextgraph_sft_models/998826_miroverse_qwen3_8b_lora32_4k_merged}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export QWEN_ENABLE_THINKING=False

export ALFWORLD_AGENT_LOOP=context_graph_isolated_agent
export ALFWORLD_WORKFLOW=alfworld_graph
export ALFWORLD_DATA_VARIANT=alfworld_graph
export ALFWORLD_METHOD_LABEL="ContextGraph controller-SFT FoldGRPO"
export ALFWORLD_PROCESS_REWARD='[flat,scope,graph]'
export ALFWORLD_STRUCTURED_GRAPH_CONTROLLER=True
export ALFWORLD_CONTROLLER_OWNED_TOOL_FORMATTING=True
export ALFWORLD_CONTROLLER_ACTION_POLICY=structural
export ALFWORLD_CONTROLLER_ALLOW_PASS=False
export ALFWORLD_CONSOLIDATION_INTERVAL=5
export ALFWORLD_ENABLE_RETRIEVAL_MEMORY=True
export ALFWORLD_INJECT_GRAPH_STATE_AFTER_ACTION=True

export ALFWORLD_TOTAL_STEPS=${ALFWORLD_TOTAL_STEPS:-30}
export ALFWORLD_TRAIN_BATCH_SIZE=${ALFWORLD_TRAIN_BATCH_SIZE:-16}
export ALFWORLD_PPO_MINI_BATCH_SIZE=${ALFWORLD_PPO_MINI_BATCH_SIZE:-16}
export ALFWORLD_ROLLOUT_N=${ALFWORLD_ROLLOUT_N:-4}
export ALFWORLD_PROMPT_LENGTH=${ALFWORLD_PROMPT_LENGTH:-4096}
export ALFWORLD_RESPONSE_LENGTH=${ALFWORLD_RESPONSE_LENGTH:-12288}
export ALFWORLD_MAX_TOKEN_LEN_PER_GPU=${ALFWORLD_MAX_TOKEN_LEN_PER_GPU:-16384}
export ALFWORLD_MAX_TURN=${ALFWORLD_MAX_TURN:-60}
export ALFWORLD_VAL_MAX_TURN=${ALFWORLD_VAL_MAX_TURN:-60}
export ALFWORLD_TURN_MAX_NEW_TOKENS=${ALFWORLD_TURN_MAX_NEW_TOKENS:-128}
export ALFWORLD_MAX_SESSION=${ALFWORLD_MAX_SESSION:-3}
export ALFWORLD_VAL_MAX_SESSION=${ALFWORLD_VAL_MAX_SESSION:-3}
export ALFWORLD_TRAIN_MAX_SAMPLES=${ALFWORLD_TRAIN_MAX_SAMPLES:-300}
export ALFWORLD_VAL_MAX_SAMPLES=${ALFWORLD_VAL_MAX_SAMPLES:-32}
export ALFWORLD_VAL_BEFORE_TRAIN=${ALFWORLD_VAL_BEFORE_TRAIN:-True}
export ALFWORLD_TEST_FREQ=${ALFWORLD_TEST_FREQ:-5}
export ALFWORLD_SAVE_FREQ=${ALFWORLD_SAVE_FREQ:-5}
export ALFWORLD_TRAINER_RESUME_MODE=${ALFWORLD_TRAINER_RESUME_MODE:-disable}
export ALFWORLD_EXPERIMENT_NAME=${ALFWORLD_EXPERIMENT_NAME:-ctxgraph_controller_sft_alfworld_8b_4n_step${ALFWORLD_TOTAL_STEPS}_${SLURM_JOB_ID}}

cd "$PROJECT_ROOT"
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph

test -s "$MODEL_PATH/config.json" || { echo "ERROR: invalid MODEL_PATH=$MODEL_PATH"; exit 2; }
test -d "$ALFWORLD_DATA/json_2.1.1" || { echo "ERROR: ALFWorld data missing under $ALFWORLD_DATA"; exit 2; }

# The full evaluator intentionally writes only 16 train rows. Rebuild a
# disjoint train/validation dataset before RL so that evaluation does not
# silently shrink the training set.
python scripts/make_alfworld_data.py \
  --mode real \
  --n_train "$ALFWORLD_TRAIN_MAX_SAMPLES" \
  --n_val "$ALFWORLD_VAL_MAX_SAMPLES" \
  --seed 42 \
  --alfworld_data "$ALFWORLD_DATA"

exec bash scripts/train_alfworld_ctxgraph_8b_4node_30step.sh
