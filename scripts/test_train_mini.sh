#!/bin/bash
#SBATCH -J foldagent-mini
#SBATCH -o foldagent-mini.%j.out
#SBATCH -e foldagent-mini.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 00:30:00
#SBATCH -A ASC24078

# Minimal FoldAgent RL training smoke test with Qwen3-0.6B
# Tests: vllm rollout + FSDP actor + FoldGRPO on 1 GPU
# No search server needed — uses a mock search env (dummy data)
set -e

# ── Environment (matching QeRL's working idev_test.sh) ──
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate foldagent

# CUDA 12.8 from nvidia/25.3 (not the default 24.7)
export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LIBRARY_PATH}
export CPATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include:${CPATH}

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
export OPENAI_API_KEY=dummy

echo "=== Step 1: Generate dummy data ==="
python scripts/make_dummy_data.py

echo "=== Step 2: Start mock search server ==="
fuser -k 18999/tcp 2>/dev/null || true
python scripts/mock_search_server.py &
MOCK_PID=$!
sleep 3
export LOCAL_SEARCH_URL="http://127.0.0.1:18999"

echo "=== Step 3: Run mini RL training ==="
python -m scripts.train_fold \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=/work/09281/chc_1996/vista/cache/Qwen3-0.6B \
  actor_rollout_ref.rollout.prompt_length=1024 \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  data.train_files=data/dummy_train.parquet \
  data.val_files=data/dummy_test.parquet \
  data.train_batch_size=4 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=4096 \
  +actor_rollout_ref.rollout.plugin.workflow=search \
  +actor_rollout_ref.rollout.plugin.max_turn=5 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=0 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=512 \
  +actor_rollout_ref.rollout.plugin.max_session=2 \
  +actor_rollout_ref.rollout.plugin.val_max_session=2 \
  +actor_rollout_ref.rollout.plugin.session_timeout=120 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=5 \
  +actor_rollout_ref.rollout.plugin.val_response_length=2048 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=2 \
  trainer.test_freq=2 \
  trainer.save_freq=-1 \
  trainer.project_name=foldagent_test \
  trainer.experiment_name=mini_qwen3_0.6b \
  trainer.logger='["console"]'

echo "=== Training finished ==="
kill $MOCK_PID 2>/dev/null || true
