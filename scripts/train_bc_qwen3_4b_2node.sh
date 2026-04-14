#!/bin/bash
#SBATCH -J foldagent-4b
#SBATCH -o foldagent-4b.%j.out
#SBATCH -e foldagent-4b.%j.err
#SBATCH -p gh
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 12:00:00
#SBATCH -A ASC24078

# FoldAgent RL training: Qwen3-4B-Instruct on BrowseComp-Plus
# 2-node Ray cluster:
#   Node 0: Search service (Qwen3-Embedding-8B) + Ray head + training worker
#   Node 1: Ray worker + training worker
# Each node has 1 GH200 GPU (96GB HBM3)
set -e

# ── Environment ──
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
export OPENAI_API_KEY=dummy

# ── Get node info ──
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODE0=$(echo "$NODELIST" | head -1)
NODE1=$(echo "$NODELIST" | tail -1)
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')

echo "=== FoldAgent Training: Qwen3-4B-Instruct on BrowseComp-Plus ==="
echo "=== Node 0 (search + ray head + train): $NODE0 ($NODE0_IP) ==="
echo "=== Node 1 (ray worker + train): $NODE1 ==="
echo "=== $(date) ==="

# ── Step 1: Start search service on Node 0 (background) ──
echo "=== Starting search service on $NODE0 ==="
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate foldagent
  export HF_HOME=/work/09281/chc_1996/vista/cache
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
  cd $PROJECT_ROOT
  python envs/search_server.py \
    --model Qwen/Qwen3-Embedding-8B \
    --corpus Tevatron/browsecomp-plus-corpus \
    --corpus-embedding-dataset miaolu3/browsecomp-plus \
    --host 0.0.0.0 --port 8010
" &
SEARCH_PID=$!

# Wait for search service
echo "Waiting for search service on ${NODE0_IP}:8010..."
for i in $(seq 1 300); do
  if curl -s http://${NODE0_IP}:8010/search -d '{"query":"test","k":1}' -H 'Content-Type: application/json' > /dev/null 2>&1; then
    echo "Search service ready after ${i}s"
    break
  fi
  if [ $i -eq 300 ]; then
    echo "ERROR: Search service failed to start after 300s"
    kill $SEARCH_PID 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

export LOCAL_SEARCH_URL="http://${NODE0_IP}:8010"

# ── Step 2: Start Ray head on Node 0 (background) ──
echo "=== Starting Ray head on $NODE0 ==="
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate foldagent
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
  export HF_HOME=/work/09281/chc_1996/vista/cache
  ray start --head --node-ip-address=$NODE0_IP --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
" &
RAY_HEAD_PID=$!
sleep 20

# ── Step 3: Start Ray worker on Node 1 (background) ──
echo "=== Starting Ray worker on $NODE1 ==="
srun --overlap --nodes=1 --ntasks=1 -w "$NODE1" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate foldagent
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
  export HF_HOME=/work/09281/chc_1996/vista/cache
  ray start --address ${NODE0_IP}:6379 --num-cpus=70 --num-gpus=1 --block
" &
RAY_WORKER_PID=$!
sleep 20

# Verify Ray cluster has 2 nodes
export RAY_ADDRESS=${NODE0_IP}:6379
echo "=== Ray cluster status ==="
ray status || echo "WARN: ray status check failed (will retry in trainer)"

# ── Step 4: Run RL training ──
echo "=== Starting FoldGRPO training ==="
echo "=== Search URL: $LOCAL_SEARCH_URL ==="
echo "=== Ray address: $RAY_ADDRESS ==="

python -m scripts.train_fold \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  actor_rollout_ref.rollout.prompt_length=8192 \
  actor_rollout_ref.rollout.response_length=32768 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  data.train_files=data/bc_train.parquet \
  data.val_files=data/bc_test.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=8192 \
  data.max_response_length=32768 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=40960 \
  +actor_rollout_ref.rollout.plugin.workflow=search_branch \
  +actor_rollout_ref.rollout.plugin.max_turn=100 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.max_session=10 \
  +actor_rollout_ref.rollout.plugin.val_max_session=10 \
  +actor_rollout_ref.rollout.plugin.session_timeout=3600 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=32768 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope]' \
  +actor_rollout_ref.rollout.plugin.max_traj=11 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_max_turn=100 \
  +actor_rollout_ref.rollout.plugin.val_response_length=32768 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=2 \
  trainer.total_training_steps=50 \
  trainer.test_freq=10 \
  trainer.save_freq=10 \
  trainer.project_name=context-graph \
  trainer.experiment_name=qwen3_4b_instruct_browsecomp \
  trainer.logger='["console","wandb"]'

echo "=== Training finished: $(date) ==="

# ── Cleanup ──
echo "=== Stopping Ray cluster ==="
ray stop --force 2>/dev/null || true
kill $RAY_HEAD_PID $RAY_WORKER_PID $SEARCH_PID 2>/dev/null || true
