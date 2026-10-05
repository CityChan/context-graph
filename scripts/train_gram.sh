#!/usr/bin/env bash
# Run inside a configured training allocation/Ray cluster. No scheduler jobs are created here.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -s configs/gram_agent.yaml ]] || { echo 'Missing configs/gram_agent.yaml: include configs/ in the training checkout/package' >&2; exit 2; }
: "${MODEL_PATH:?Set the actor checkpoint directory}"
: "${GRAM_TRAIN_DATA:?Set prepared train/data.parquet}"
: "${GRAM_VAL_DATA:?Set prepared validation/data.parquet}"
: "${GRAM_MEMORY_ENDPOINT:?Set a separately served frozen helper endpoint}"
: "${GRAM_MEMORY_MODEL:?Set the helper served model name}"
: "${GRAM_MEMORY_REVISION:?Set the frozen helper checkpoint revision}"
: "${GRAM_FROZEN_MEMORY:?Set true only for a separate frozen helper service}"
[[ "$GRAM_FROZEN_MEMORY" == true ]] || { echo 'GRAM_FROZEN_MEMORY must be true' >&2; exit 2; }
[[ "$GRAM_TRAIN_DATA" != "$GRAM_VAL_DATA" ]] || { echo 'Train and validation files must differ' >&2; exit 2; }
python -m scripts.train_gram --check-data "$GRAM_TRAIN_DATA" "$GRAM_VAL_DATA"
CONTEXT_LENGTH=${GRAM_CONTEXT_LENGTH:-32768}
STEP_TOKENS=${GRAM_STEP_TOKENS:-2048}
PROMPT_LENGTH=$((CONTEXT_LENGTH - STEP_TOKENS - 32))
RESPONSE_LENGTH=$((STEP_TOKENS + 32))
(( PROMPT_LENGTH > 0 )) || { echo 'Invalid context budget' >&2; exit 2; }
python -m scripts.train_gram \
  algorithm.adv_estimator=foldgrpo \
  algorithm.foldgrpo_process_reward_mode=relative_extrema \
  algorithm.fix_bad_positive_adv=False algorithm.use_kl_in_reward=False \
  actor_rollout_ref.rollout.agent.default_agent_loop=gram_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=configs/gram_agent.yaml \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GRAM_TP:-1}" \
  actor_rollout_ref.rollout.n=5 actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.prompt_length="$PROMPT_LENGTH" \
  actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  data.train_files="$GRAM_TRAIN_DATA" data.val_files="$GRAM_VAL_DATA" \
  data.train_batch_size=8 data.return_raw_chat=True \
  data.max_prompt_length="$PROMPT_LENGTH" data.max_response_length="$RESPONSE_LENGTH" \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.qwen_enable_thinking=False \
  +actor_rollout_ref.rollout.plugin.gram.memory_endpoint="$GRAM_MEMORY_ENDPOINT" \
  +actor_rollout_ref.rollout.plugin.gram.memory_model="$GRAM_MEMORY_MODEL" \
  +actor_rollout_ref.rollout.plugin.gram.memory_revision="$GRAM_MEMORY_REVISION" \
  +actor_rollout_ref.rollout.plugin.gram.frozen_memory_acknowledged=True \
  +actor_rollout_ref.rollout.plugin.gram.episode.max_steps="${GRAM_MAX_STEPS:-64}" \
  +actor_rollout_ref.rollout.plugin.gram.episode.max_step_tokens="$STEP_TOKENS" \
  +actor_rollout_ref.rollout.plugin.gram.episode.max_episode_tokens="${GRAM_EPISODE_TOKENS:-32768}" \
  trainer.nnodes="${GRAM_NNODES:-1}" trainer.n_gpus_per_node="${GRAM_GPUS_PER_NODE:-1}" \
  trainer.project_name=gram trainer.experiment_name=gram-document-stream \
  trainer.logger=console trainer.total_training_steps="${GRAM_TRAINING_STEPS:-50}" \
  "$@"
