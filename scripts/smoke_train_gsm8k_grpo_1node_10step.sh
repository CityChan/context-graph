#!/bin/bash

# Minimal rule-reward GRPO benchmark for one four-GPU Vista GH200 node.
# Run this inside an active idev allocation; no search server or API key is used.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
GSM8K_DATA_DIR=${GSM8K_DATA_DIR:-${SCRATCH:?SCRATCH must be set}/context-graph-data/gsm8k}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
TS=$(date +%Y%m%d_%H%M%S)
RUN_TAG=${RUN_TAG:-gsm8k-qwen25-1p5b-grpo-10step-fixed}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH/context-graph-ckpts/$RUN_TAG-$TS}
RUN_LOG=${RUN_LOG:-$PROJECT_ROOT/logs/$RUN_TAG-$TS.log}

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this inside an active Vista idev allocation."
  exit 2
fi

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"
mkdir -p logs "$GSM8K_DATA_DIR" "$CHECKPOINT_ROOT"

VISIBLE_GPUS=$(nvidia-smi -L | wc -l)
NUM_GPUS=${NUM_GPUS:-$VISIBLE_GPUS}
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "ERROR: no GPU is visible on $(hostname)."
  exit 2
fi
if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
  echo "ERROR: requested $NUM_GPUS GPUs but only $VISIBLE_GPUS are visible on $(hostname)."
  exit 2
fi

export HF_HOME HF_HUB_CACHE TOKENIZERS_PARALLELISM=false
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE RAY_ADDRESS

if [ ! -s "$GSM8K_DATA_DIR/train.parquet" ] || [ ! -s "$GSM8K_DATA_DIR/test.parquet" ]; then
  python scripts/prepare_gsm8k_grpo_data.py --output-dir "$GSM8K_DATA_DIR"
fi

echo "GSM8K GRPO smoke: model=$MODEL_PATH gpus=$NUM_GPUS steps=$TOTAL_TRAINING_STEPS"
echo "log=$RUN_LOG"
echo "checkpoints=$CHECKPOINT_ROOT"

set +e
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  data.train_files="$GSM8K_DATA_DIR/train.parquet" \
  data.val_files="$GSM8K_DATA_DIR/test.parquet" \
  data.train_max_samples=512 \
  data.val_max_samples=256 \
  data.train_batch_size=16 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.return_raw_chat=True \
  reward_manager.name=naive \
  trainer.val_before_train=True \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq=5 \
  trainer.save_freq=10 \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.project_name=context-graph \
  trainer.experiment_name="$RUN_TAG-$TS" \
  trainer.logger=console \
  trainer.default_local_dir="$CHECKPOINT_ROOT" \
  trainer.resume_mode=disable 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
  echo "GSM8K GRPO smoke failed with exit code $RC; log=$RUN_LOG"
  exit "$RC"
fi

echo "GSM8K GRPO smoke completed; log=$RUN_LOG"
