#!/bin/bash
#SBATCH -J both-alf30h
#SBATCH -o both-alf30h.%j.out
#SBATCH -e both-alf30h.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A ASC24078

# Qwen3-30B-A3B (MoE) on ALFWorld hard mode (no admissible commands), 8 GH200 nodes
# Runs both FoldAgent and ContextGraph sequentially, 20 steps each
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
export OPENAI_API_KEY=REDACTED_OPENAI_KEY
export WANDB_API_KEY=REDACTED_WANDB_KEY

# ── Node discovery ──
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

echo "=== Qwen3-30B-A3B on ALFWorld HARD, ${NUM_NODES}-node Ray cluster ==="
echo "=== Head: $NODE0 ($NODE0_IP) ==="
echo "=== $(date) ==="

# ── Generate data (hard mode: no admissible commands) ──
python scripts/make_alfworld_data.py --mode mock --hard --n_train 300 --n_val 80

# ── Start Ray head on Node 0 ──
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

# ── Start Ray workers on Nodes 1..N-1 ──
WORKER_PIDS=()
for i in $(seq 1 $((NUM_NODES-1))); do
  WORKER_NODE=${NODELIST[$i]}
  echo "=== Starting Ray worker on $WORKER_NODE (node $i) ==="
  srun --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c "
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate foldagent
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
    export HF_HOME=/work/09281/chc_1996/vista/cache
    ray start --address ${NODE0_IP}:6379 --num-cpus=70 --num-gpus=1 --block
  " &
  WORKER_PIDS+=($!)
  sleep 5
done
sleep 20

export RAY_ADDRESS=${NODE0_IP}:6379
echo "=== Ray cluster status ==="
ray status || echo "WARN: ray status check failed"

# ── Common training args for 30B MoE ──
# FSDP shards 30B params (7.5GB/GPU) + grads + optim state (30GB/GPU) across 8 GPUs
# vLLM TP=1 (each GH200 has 1 GPU), gpu_memory_utilization tuned for MoE weight size
COMMON_ARGS="
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Instruct-2507 \
  actor_rollout_ref.rollout.prompt_length=4096 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  data.train_batch_size=16 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=8192 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.max_turn=30 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=512 \
  +actor_rollout_ref.rollout.plugin.max_session=3 \
  +actor_rollout_ref.rollout.plugin.val_max_session=3 \
  +actor_rollout_ref.rollout.plugin.session_timeout=600 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=2048 \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=30 \
  +actor_rollout_ref.rollout.plugin.val_response_length=4096 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=20 \
  trainer.test_freq=10 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.logger=[\"console\",\"wandb\"]
"

# ── RUN 1: FoldAgent ──
echo "=== [RUN 1/2] FoldAgent on ALFWorld HARD ==="
python -m scripts.train_fold \
  $COMMON_ARGS \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  data.train_files=data/alfworld_train.parquet \
  data.val_files=data/alfworld_test.parquet \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld \
  +actor_rollout_ref.rollout.plugin.process_reward=none \
  trainer.experiment_name=fold_alfworld_30b_hard_20

echo "=== [RUN 1/2] FoldAgent finished: $(date) ==="
sleep 30

# ── RUN 2: ContextGraph ISOLATED ──
echo "=== [RUN 2/2] ContextGraph ISOLATED on ALFWorld HARD ==="
python -m scripts.train_graph \
  $COMMON_ARGS \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  data.train_files=data/alfworld_graph_train.parquet \
  data.val_files=data/alfworld_graph_test.parquet \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld_graph \
  +actor_rollout_ref.rollout.plugin.process_reward=graph \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
  trainer.experiment_name=ctxgraph_iso_alfworld_30b_hard_20

echo "=== [RUN 2/2] ContextGraph finished: $(date) ==="
echo "=== ALL DONE: $(date) ==="

# Cleanup
kill $RAY_HEAD_PID 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do
  kill $pid 2>/dev/null || true
done
