#!/bin/bash
#SBATCH -J test-alf-4b
#SBATCH -o logs/test-alf-4b.%j.out
#SBATCH -e logs/test-alf-4b.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 01:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# Smoke test: Qwen3-4B-Instruct-2507 on ALFWorld, 1 node / 1 hour
# Goal: validate environment + 4B model load + 3 RL steps on a single GH200.
# Reduced settings (response=8192, batch=8, n=4, 3 steps) — NOT a real run.
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
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell before running this script}"

# ── Node info ──
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

echo "════════════════════════════════════════════════════════════════"
echo "  SMOKE TEST: Qwen3-4B-Instruct-2507 on ALFWorld"
echo "  Nodes: $NUM_NODES   Head: $NODE0 ($NODE0_IP)   Time: $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── Sanity: Python / CUDA / project layout ──
echo "--- Python: $(python --version 2>&1) ---"
echo "--- CUDA: $(nvcc --version | tail -1) ---"
echo "--- conda env: $CONDA_DEFAULT_ENV ---"
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)" || echo "WARN: vllm import failed"
python -c "import verl; print('verl OK')" || echo "WARN: verl import failed"
python -c "import textworld; import alfworld; print('textworld + alfworld OK')" || echo "WARN: alfworld import failed"
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)" || echo "WARN: flash_attn import failed"

# ── Generate ALFWorld data (small) ──
echo "--- Generating ALFWorld parquet ---"
python scripts/make_alfworld_data.py --hard --n_train 32 --n_val 8

# ── Ray head on Node 0 ──
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
sleep 20

export RAY_ADDRESS=${NODE0_IP}:6379
echo "--- Ray cluster status ---"
ray status || echo "WARN: ray status check failed"

# ── Run 3 RL steps just to prove the pipeline works ──
echo "════════════════════════════════════════════════════════════════"
echo "  Launching FoldGRPO smoke test (3 steps, batch=8, n=4)"
echo "════════════════════════════════════════════════════════════════"

python -m scripts.train_fold \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  actor_rollout_ref.rollout.prompt_length=4096 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  data.train_files=data/alfworld_train.parquet \
  data.val_files=data/alfworld_test.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=4096 \
  data.max_response_length=8192 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=12288 \
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
  +actor_rollout_ref.rollout.plugin.val_response_length=8192 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=3 \
  trainer.test_freq=999 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.experiment_name=smoke_alfworld_4b_1n \
  trainer.logger='["console","wandb"]'

RC=$?
echo "════════════════════════════════════════════════════════════════"
if [ $RC -eq 0 ]; then
  echo "  ✓ SMOKE TEST PASSED — environment is ready for production run"
else
  echo "  ✗ SMOKE TEST FAILED (exit $RC) — fix before launching larger job"
fi
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── Cleanup ──
kill $RAY_HEAD_PID 2>/dev/null || true

exit $RC
