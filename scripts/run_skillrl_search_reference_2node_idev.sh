#!/bin/bash

# Two-node Vista diagnostic for the official SkillRL Search environment.
# Node 0 hosts the 64 GB FAISS index; node 1 uses four GPUs for eval/training.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
REFERENCE_ROOT=${REFERENCE_ROOT:-${SCRATCH:?SCRATCH must be set}/skillrl_search_reference}
SKILLRL_ROOT=${SKILLRL_ROOT:-$REFERENCE_ROOT/SkillRL}
ENV_NAME=${ENV_NAME:-skillrl_search}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
RUN_MODE=${RUN_MODE:-eval_ladder}
TRAIN_STEPS=${TRAIN_STEPS:-10}
PORT=${PORT:-8030}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

MODEL_ROOT=$REFERENCE_ROOT/models
DATA_ROOT=$REFERENCE_ROOT/data
RETRIEVER_ROOT=$REFERENCE_ROOT/retriever
QWEN_MODEL=$MODEL_ROOT/Qwen2.5-7B-Instruct
SFT_MODEL=$MODEL_ROOT/Search-7B-SFT
RL_MODEL=$MODEL_ROOT/Search-7B-RL-HF
E5_MODEL=$MODEL_ROOT/e5-base-v2
LOG_ROOT=$REFERENCE_ROOT/logs/$RUN_TAG
CHECKPOINT_ROOT=$REFERENCE_ROOT/checkpoints/$RUN_TAG

activate_reference_env() {
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  export HF_HOME TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
}

run_retriever() {
  activate_reference_env
  cd "$SKILLRL_ROOT"
  exec python examples/search/retriever/retrieval_server.py --index_path "$RETRIEVER_ROOT/e5_Flat.index" --corpus_path "$RETRIEVER_ROOT/wiki-18.jsonl" --topk 3 --retriever_name e5 --retriever_model "$E5_MODEL" --faiss_gpu --port "$PORT"
}

run_trainer() {
  local model_path=$1
  local use_skills=$2
  local val_only=$3
  local experiment=$4
  local total_steps=$5
  activate_reference_env
  cd "$SKILLRL_ROOT"
  ray stop --force >/dev/null 2>&1 || true
  unset RAY_ADDRESS
  export MODEL_PATH=$model_path
  bash examples/grpo_trainer/run_search_skills.sh vllm data.train_files="$DATA_ROOT/train_diag.parquet" data.val_files="$DATA_ROOT/test_diag.parquet" data.train_batch_size=16 data.val_batch_size=448 data.max_prompt_length=6000 data.max_response_length=1024 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=128 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 env.rollout.n=8 env.search.search_url="http://$RETRIEVER_NODE:$PORT/retrieve" env.use_skills_only_memory="$use_skills" trainer.logger="['console']" trainer.experiment_name="$experiment" trainer.default_local_dir="$CHECKPOINT_ROOT/$experiment" trainer.val_only="$val_only" trainer.total_epochs=1 trainer.total_training_steps="$total_steps" trainer.test_freq=5 trainer.save_freq=1 trainer.val_before_train=True
}

if [ "${1:-}" = retriever ]; then
  run_retriever
  exit 0
fi
if [ "${1:-}" = trainer ]; then
  shift
  run_trainer "$@"
  exit 0
fi

if [ -z "${SLURM_NODELIST:-}" ]; then
  echo "ERROR: no active Slurm allocation; start a fresh two-node idev first."
  exit 2
fi
mapfile -t IDEV_NODES < <(scontrol show hostnames "$SLURM_NODELIST")
if [ "${#IDEV_NODES[@]}" -lt 2 ]; then
  echo "ERROR: expected at least two idev nodes, got ${#IDEV_NODES[@]}."
  exit 2
fi

RETRIEVER_NODE=${IDEV_NODES[0]}
TRAINER_NODE=${IDEV_NODES[1]}
export PROJECT_ROOT REFERENCE_ROOT SKILLRL_ROOT ENV_NAME HF_HOME RUN_MODE TRAIN_STEPS PORT RUN_TAG RETRIEVER_NODE
mkdir -p "$LOG_ROOT" "$CHECKPOINT_ROOT"

for required in "$QWEN_MODEL/config.json" "$SFT_MODEL/config.json" "$E5_MODEL/config.json" "$RETRIEVER_ROOT/e5_Flat.index" "$RETRIEVER_ROOT/wiki-18.jsonl" "$DATA_ROOT/train_diag.parquet" "$DATA_ROOT/test_diag.parquet"; do
  if [ ! -s "$required" ]; then
    echo "ERROR: missing $required; run scripts/prepare_skillrl_search_reference_vista.sh on a login node."
    exit 2
  fi
done

echo "RUN_TAG=$RUN_TAG"
echo "retriever_node=$RETRIEVER_NODE trainer_node=$TRAINER_NODE mode=$RUN_MODE"
srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 -w "$RETRIEVER_NODE" bash "$PROJECT_ROOT/scripts/run_skillrl_search_reference_2node_idev.sh" retriever > "$LOG_ROOT/retriever.log" 2>&1 &
RETRIEVER_JOB_PID=$!
cleanup() {
  kill "$RETRIEVER_JOB_PID" >/dev/null 2>&1 || true
  wait "$RETRIEVER_JOB_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 90); do
  if curl --silent --fail --max-time 10 -X POST "http://$RETRIEVER_NODE:$PORT/retrieve" -H "Content-Type: application/json" -d '{"query":"Eiffel Tower","topk":1,"return_scores":false}' >/dev/null; then
    READY=1
    break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: retriever failed readiness; inspect $LOG_ROOT/retriever.log"
  exit 1
fi

run_one() {
  local model_path=$1
  local use_skills=$2
  local val_only=$3
  local experiment=$4
  local total_steps=$5
  local log_path=$LOG_ROOT/$experiment.log
  set +e
  srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=4 -w "$TRAINER_NODE" bash "$PROJECT_ROOT/scripts/run_skillrl_search_reference_2node_idev.sh" trainer "$model_path" "$use_skills" "$val_only" "$experiment" "$total_steps" 2>&1 | tee "$log_path"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: $experiment exited $rc; inspect $log_path"
    exit "$rc"
  fi
  if [ "$val_only" = false ]; then
    python "$PROJECT_ROOT/scripts/audit_skillrl_search_reference.py" --require-all-benchmarks --require-training-health "$log_path"
  else
    python "$PROJECT_ROOT/scripts/audit_skillrl_search_reference.py" --require-all-benchmarks "$log_path"
  fi
}

case "$RUN_MODE" in
  eval_ladder)
    run_one "$QWEN_MODEL" false true qwen25_7b_instruct_eval 1
    run_one "$SFT_MODEL" true true search_7b_sft_eval 1
    if [ -s "$RL_MODEL/config.json" ]; then
      run_one "$RL_MODEL" true true search_7b_rl_eval 1
    else
      echo "RL checkpoint not present; rerun preparation with DOWNLOAD_RL=1 to add the third rung."
    fi
    ;;
  train_smoke)
    run_one "$SFT_MODEL" true false search_7b_sft_grpo_${TRAIN_STEPS}step "$TRAIN_STEPS"
    checkpoint=$CHECKPOINT_ROOT/search_7b_sft_grpo_${TRAIN_STEPS}step/global_step_${TRAIN_STEPS}/actor
    if [ ! -d "$checkpoint" ]; then
      echo "ERROR: expected checkpoint missing: $checkpoint"
      exit 1
    fi
    echo "GRPO smoke passed: finite logged health metrics and checkpoint $checkpoint"
    ;;
  *)
    echo "ERROR: RUN_MODE must be eval_ladder or train_smoke, got $RUN_MODE"
    exit 2
    ;;
esac
