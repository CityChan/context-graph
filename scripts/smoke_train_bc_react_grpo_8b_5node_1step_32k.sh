#!/bin/bash
#SBATCH -J smoke-bc-8b-react-grpo
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/smoke-bc-8b-react-grpo.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/smoke-bc-8b-react-grpo.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# One real standard-GRPO optimizer step through the production ReAct baseline.
# It preserves the 8K prompt + 32K response configuration while reducing the
# smoke batch to four prompts x two rollouts.
set -euo pipefail

export EXPECTED_NUM_NODES=5
export RUN_TAG=react_grpo_smoke_5n_1step_32k
export ADV_ESTIMATOR=grpo
export PROMPT_LENGTH=8192
export RESPONSE_LENGTH=32768
export CONTEXT_LENGTH=40960
export TRAIN_BATCH_SIZE=4
export PPO_MINI_BATCH_SIZE=2
export ROLLOUT_N=2
export TRAIN_LR=1e-6
export SESSION_TIMEOUT=3600
export TOTAL_TRAINING_STEPS=1
export VAL_BEFORE_TRAIN=False
export TEST_FREQ=0
export SAVE_FREQ=0

SUBMIT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
if [[ -f "$SUBMIT_DIR/scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh" ]]; then
  BASE_SCRIPT="$SUBMIT_DIR/scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh"
elif [[ -f "$SUBMIT_DIR/train_bc_baseline_8b_4node_24h_v3_32k.sh" ]]; then
  BASE_SCRIPT="$SUBMIT_DIR/train_bc_baseline_8b_4node_24h_v3_32k.sh"
else
  echo "ERROR: cannot find train_bc_baseline_8b_4node_24h_v3_32k.sh under SLURM_SUBMIT_DIR=$SUBMIT_DIR" >&2
  exit 1
fi

exec bash "$BASE_SCRIPT"
