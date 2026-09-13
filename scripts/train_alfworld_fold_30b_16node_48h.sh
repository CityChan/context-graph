#!/bin/bash
#SBATCH -J fold-alf-30b-16n
#SBATCH -o logs/fold-alf-30b-16n.%j.out
#SBATCH -e logs/fold-alf-30b-16n.%j.err
#SBATCH -p gh
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# Production: FoldAgent (Qwen3-30B-A3B-Thinking-2507) on ALFWorld
#   16 nodes × 1 GH200 = 16 GPUs (FSDP)  |  48 hours
# Hyperparams adapted from Sun et al. 2025 (FoldAgent paper, Slime → verl):
#   - context window  : 32K (response_length)
#   - branch threshold: 8K  (branch_len ≈ paper's "context penalty threshold")
#   - lr 5e-6, weight_decay 0.1, KL coef 0.001
#   - batch 32 prompts × n=8 GRPO group = 256 trajectories per rollout step
# Slime-only features NOT replicated: INT4 quantization, QAT, TIS clip 2.0,
#   token-in-token-out (verl tokenizes between turns).
# ─────────────────────────────────────────────────────────────────────
set -e

# ── Environment ──
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph

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

# ── Node info ──
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

echo "════════════════════════════════════════════════════════════════"
echo "  PRODUCTION: FoldAgent / Qwen3-30B-A3B-Thinking-2507 / ALFWorld"
echo "  Nodes: $NUM_NODES   Head: $NODE0 ($NODE0_IP)"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"
for i in $(seq 0 $((NUM_NODES-1))); do
  echo "    Node $i: ${NODELIST[$i]}"
done

# ── Generate ALFWorld data ──
echo "--- Generating ALFWorld parquet ---"
python scripts/make_alfworld_data.py --n_train 1000 --n_val 100

# ── Ray head ──
echo "--- Starting Ray head on $NODE0 ---"
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
  export HF_HOME=/work/09281/chc_1996/vista/cache
  ray start --head --node-ip-address=$NODE0_IP --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
" &
RAY_HEAD_PID=$!
sleep 30

# ── Ray workers ──
WORKER_PIDS=()
for i in $(seq 1 $((NUM_NODES-1))); do
  WORKER_NODE=${NODELIST[$i]}
  echo "--- Starting Ray worker on $WORKER_NODE (node $i) ---"
  srun --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c "
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
    export HF_HOME=/work/09281/chc_1996/vista/cache
    ray start --address ${NODE0_IP}:6379 --num-cpus=70 --num-gpus=1 --block
  " &
  WORKER_PIDS+=($!)
  sleep 5
done
sleep 30

export RAY_ADDRESS=${NODE0_IP}:6379
echo "--- Ray cluster status ---"
ray status || echo "WARN: ray status check failed"

# ── RL training ──
echo "════════════════════════════════════════════════════════════════"
echo "  Launching FoldGRPO training (target 500 steps, ~48h budget)"
echo "════════════════════════════════════════════════════════════════"

python -m scripts.train_fold \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Thinking-2507 \
  actor_rollout_ref.rollout.prompt_length=8192 \
  actor_rollout_ref.rollout.response_length=32768 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  data.train_files=data/alfworld_train.parquet \
  data.val_files=data/alfworld_test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=8192 \
  data.max_response_length=32768 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=40960 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld \
  +actor_rollout_ref.rollout.plugin.max_turn=50 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.max_session=10 \
  +actor_rollout_ref.rollout.plugin.val_max_session=10 \
  +actor_rollout_ref.rollout.plugin.session_timeout=1800 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=8192 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope]' \
  +actor_rollout_ref.rollout.plugin.max_traj=11 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=50 \
  +actor_rollout_ref.rollout.plugin.val_response_length=32768 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=500 \
  trainer.test_freq=25 \
  trainer.save_freq=100 \
  trainer.project_name=context-graph \
  trainer.experiment_name=fold_alfworld_30b_${NUM_NODES}n \
  trainer.logger='["console","wandb"]'

echo "════════════════════════════════════════════════════════════════"
echo "  Training finished: $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── Cleanup ──
kill $RAY_HEAD_PID 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do
  kill $pid 2>/dev/null || true
done
