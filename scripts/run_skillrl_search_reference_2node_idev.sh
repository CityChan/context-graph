#!/bin/bash

# Two-node Vista evaluation diagnostic for the official SkillRL Search
# environment. Node 0 hosts the 64 GB FAISS index on one GH200; node 1 runs
# validation on one GH200. GRPO training needs a separate five-node allocation.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
REFERENCE_ROOT=${REFERENCE_ROOT:-${SCRATCH:?SCRATCH must be set}/skillrl_search_reference}
SKILLRL_ROOT=${SKILLRL_ROOT:-$REFERENCE_ROOT/SkillRL}
ENV_NAME=${ENV_NAME:-skillrl_search}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
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
  local experiment=$3
  activate_reference_env
  cd "$SKILLRL_ROOT"
  ray stop --force >/dev/null 2>&1 || true
  unset RAY_ADDRESS
  export MODEL_PATH=$model_path
  bash examples/grpo_trainer/run_search_skills.sh vllm data.train_files="$DATA_ROOT/train_diag.parquet" data.val_files="$DATA_ROOT/test_diag.parquet" data.train_batch_size=16 data.val_batch_size=448 data.max_prompt_length=6000 data.max_response_length=1024 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=128 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 env.rollout.n=8 env.search.search_url="http://$RETRIEVER_NODE:$PORT/retrieve" env.use_skills_only_memory="$use_skills" trainer.logger="['console']" trainer.experiment_name="$experiment" trainer.default_local_dir="$CHECKPOINT_ROOT/$experiment" trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.val_only=True trainer.total_epochs=1 trainer.total_training_steps=1 trainer.test_freq=1 trainer.save_freq=-1 trainer.val_before_train=True
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
export PROJECT_ROOT REFERENCE_ROOT SKILLRL_ROOT ENV_NAME HF_HOME PORT RUN_TAG RETRIEVER_NODE
mkdir -p "$LOG_ROOT" "$CHECKPOINT_ROOT"

for required in "$QWEN_MODEL/config.json" "$SFT_MODEL/config.json" "$E5_MODEL/config.json" "$RETRIEVER_ROOT/e5_Flat.index" "$RETRIEVER_ROOT/wiki-18.jsonl" "$DATA_ROOT/train_diag.parquet" "$DATA_ROOT/test_diag.parquet"; do
  if [ ! -s "$required" ]; then
    echo "ERROR: missing $required; submit scripts/prepare_skillrl_search_reference_vista_batch.sh first."
    exit 2
  fi
done

echo "RUN_TAG=$RUN_TAG"
echo "retriever_node=$RETRIEVER_NODE evaluation_node=$TRAINER_NODE"
srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=1 -w "$RETRIEVER_NODE" bash "$PROJECT_ROOT/scripts/run_skillrl_search_reference_2node_idev.sh" retriever > "$LOG_ROOT/retriever.log" 2>&1 &
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
  local experiment=$3
  local log_path=$LOG_ROOT/$experiment.log
  set +e
  srun --overlap --nodes=1 --ntasks=1 --gpus-per-node=1 -w "$TRAINER_NODE" bash "$PROJECT_ROOT/scripts/run_skillrl_search_reference_2node_idev.sh" trainer "$model_path" "$use_skills" "$experiment" 2>&1 | tee "$log_path"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: $experiment exited $rc; inspect $log_path"
    exit "$rc"
  fi
  python "$PROJECT_ROOT/scripts/audit_skillrl_search_reference.py" --require-all-benchmarks "$log_path"
}

run_one "$QWEN_MODEL" false qwen25_7b_instruct_eval
run_one "$SFT_MODEL" true search_7b_sft_eval
if [ -s "$RL_MODEL/config.json" ]; then
  run_one "$RL_MODEL" true search_7b_rl_eval
else
  echo "RL checkpoint not present; resubmit preparation with DOWNLOAD_RL=1 to add the third rung."
fi
