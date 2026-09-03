#!/bin/bash
# One-step end-to-end GraphRPO mechanics smoke on a five-node Vista allocation.
# The deterministic evaluator started here is not valid for scientific runs.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
: "${MODEL_PATH:?Set MODEL_PATH to the merged Hugging Face SFT checkpoint}"

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#NODELIST[@]}" -ne 5 ]; then
  echo "ERROR: GraphRPO smoke requires exactly five allocated nodes; got ${#NODELIST[@]}"
  exit 1
fi
EVALUATOR_NODE=${NODELIST[0]}
EVALUATOR_NODE_IP=$(getent hosts "$EVALUATOR_NODE" | awk '{print $1}')
EVALUATOR_PORT=${GRAPH_RPO_SMOKE_EVALUATOR_PORT:-19001}
EVALUATOR_LOG="$PROJECT_ROOT/logs/graphrpo-smoke-evaluator-${SLURM_JOB_ID:-idev}.log"
mkdir -p "$PROJECT_ROOT/logs"

srun --overlap --nodes=1 --ntasks=1 -w "$EVALUATOR_NODE" bash -c "source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh; conda activate cxtgraph; cd '$PROJECT_ROOT'; export PYTHONPATH='$PROJECT_ROOT':\${PYTHONPATH:-}; exec python -u scripts/serve_graph_evaluator_smoke.py --port '$EVALUATOR_PORT'" >"$EVALUATOR_LOG" 2>&1 &
EVALUATOR_PID=$!
cleanup_smoke_evaluator() {
  kill "$EVALUATOR_PID" 2>/dev/null || true
}
trap cleanup_smoke_evaluator EXIT

EVALUATOR_READY=0
for _ in $(seq 1 120); do
  if curl --noproxy '*' -fsS "http://${EVALUATOR_NODE_IP}:${EVALUATOR_PORT}/health" >/dev/null 2>&1; then
    EVALUATOR_READY=1
    break
  fi
  if ! kill -0 "$EVALUATOR_PID" 2>/dev/null; then
    echo "ERROR: smoke evaluator exited before becoming healthy"
    tail -80 "$EVALUATOR_LOG" || true
    exit 1
  fi
  sleep 1
done
if [ "$EVALUATOR_READY" != "1" ]; then
  echo "ERROR: smoke evaluator did not become healthy"
  tail -80 "$EVALUATOR_LOG" || true
  exit 1
fi

export GRAPH_RPO_EVALUATOR_URL="http://${EVALUATOR_NODE_IP}:${EVALUATOR_PORT}/score"
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export ROLLOUT_N=${ROLLOUT_N:-2}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-4}
export VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-4}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export TEST_FREQ=${TEST_FREQ:-0}
export SAVE_FREQ=${SAVE_FREQ:-1}
export MAX_SESSION=${MAX_SESSION:-3}
export VAL_MAX_SESSION=${VAL_MAX_SESSION:-3}
export MAX_TURN=${MAX_TURN:-40}
export RUN_TAG=${RUN_TAG:-graphrpo_sft_1step_smoke}
export GRAPH_RPO_SERIALIZATION_PENALTY=${GRAPH_RPO_SERIALIZATION_PENALTY:-0.02}
export BC_DISABLE_WANDB=${BC_DISABLE_WANDB:-0}

echo "WARNING: using deterministic smoke-only graph evaluator at $GRAPH_RPO_EVALUATOR_URL"
bash scripts/train_bc_ctxgraph_8b_graphrpo_5node_48h.sh
