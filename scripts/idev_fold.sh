#!/bin/bash
# Run FoldAgent on ALFWorld inside idev session (NO sbatch header)
# Usage (from idev shell):
#   cd /work/09281/chc_1996/vista/context-graph
#   bash scripts/idev_fold.sh
set -e

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate foldagent

export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LIBRARY_PATH}
export CPATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/include:${CPATH}

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCHDYNAMO_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"

export HF_HOME=/work/09281/chc_1996/vista/cache
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY in your shell before running this script}"
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell before running this script}"

echo "=== [IDEV] FoldAgent on ALFWorld (4B Instruct, 20 steps) ==="
echo "=== Node: $(hostname), $(date) ==="

python scripts/make_alfworld_data.py --n_train 300 --n_val 80

python -m scripts.train_fold \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  actor_rollout_ref.rollout.prompt_length=4096 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  data.train_files=data/alfworld_train.parquet \
  data.val_files=data/alfworld_test.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=8192 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld \
  +actor_rollout_ref.rollout.plugin.max_turn=20 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=512 \
  +actor_rollout_ref.rollout.plugin.max_session=3 \
  +actor_rollout_ref.rollout.plugin.val_max_session=3 \
  +actor_rollout_ref.rollout.plugin.session_timeout=300 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=2048 \
  +actor_rollout_ref.rollout.plugin.process_reward=none \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=20 \
  +actor_rollout_ref.rollout.plugin.val_response_length=4096 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=20 \
  trainer.test_freq=10 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.experiment_name=fold_alfworld_4b_hard_20 \
  trainer.logger='["console","wandb"]'

echo "=== [IDEV] FoldAgent finished: $(date) ==="
