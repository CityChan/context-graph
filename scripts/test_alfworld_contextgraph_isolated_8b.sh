#!/bin/bash
#SBATCH -J ctxg-alf8b
#SBATCH -o ctxg-alf8b.%j.out
#SBATCH -e ctxg-alf8b.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 06:00:00
#SBATCH -A ASC24078

# ContextGraph (ISOLATED) on ALFWorld, 8B model, 8 nodes
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

# ── Get node info ──
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODES=($NODELIST)
NODE0=${NODES[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')

echo "=== TEST: ContextGraph ISOLATED on ALFWorld, 8B, 8 nodes ==="
echo "=== Nodes: $NODELIST ==="
echo "=== Head: $NODE0 ($NODE0_IP) ==="
echo "=== $(date) ==="

# ── Step 1: Generate data ──
python scripts/make_alfworld_data.py --mode real --n_train 300 --n_val 80

# ── Step 2: Start Ray head on Node 0 ──
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

# ── Step 3: Start Ray workers on Nodes 1-7 ──
for i in $(seq 1 7); do
  echo "=== Starting Ray worker on ${NODES[$i]} ==="
  srun --overlap --nodes=1 --ntasks=1 -w "${NODES[$i]}" bash -c "
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate foldagent
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
    export HF_HOME=/work/09281/chc_1996/vista/cache
    ray start --address ${NODE0_IP}:6379 --num-cpus=70 --num-gpus=1 --block
  " &
done
sleep 30

export RAY_ADDRESS=${NODE0_IP}:6379
echo "=== Ray cluster status ==="
ray status || echo "WARN: ray status check failed"

# ── Step 4: Train ──
echo "=== Starting ContextGraph (ISOLATED) + FoldGRPO on ALFWorld (5 steps, 8B, 8 nodes) ==="

python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.rollout.prompt_length=4096 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  data.train_files=data/alfworld_graph_train.parquet \
  data.val_files=data/alfworld_graph_test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=4096 \
  data.max_response_length=4096 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=8192 \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld_graph \
  +actor_rollout_ref.rollout.plugin.max_turn=20 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=256 \
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
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=8 \
  trainer.total_training_steps=50 \
  trainer.test_freq=10 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.experiment_name=ctxgraph_iso_alfworld_8b \
  trainer.logger='["console","wandb"]'

echo "=== Test finished: $(date) ==="
