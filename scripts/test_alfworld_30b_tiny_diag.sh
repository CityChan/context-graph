#!/bin/bash
#SBATCH -J test-alf-30b-bf16-8n
#SBATCH -o test-alf-30b-bf16-8n.%j.out
#SBATCH -e test-alf-30b-bf16-8n.%j.err
#SBATCH -p gh-dev
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 02:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# Smoke test: Qwen3-30B-A3B-Thinking-2507 on ALFWorld, 8 nodes / 2 hours, BF16
# Goal: validate environment + Ray cluster + 30B model load + 2-3 RL steps.
# Reduced settings (response=8192, batch=8, n=4, 3 steps) — NOT a real run.
# ─────────────────────────────────────────────────────────────────────
set -e
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export HF_HUB_DISABLE_FILE_LOCKING=1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
# expandable_segments incompatible with vLLM sleep_mode; disabled
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source $WORK/.wandb_env

# ── Environment ──
source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
  export PATH=${CONDA_PREFIX}/bin:${PATH}; hash -r

export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LIBRARY_PATH}
export CPATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/include:${CPATH}

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCHDYNAMO_DISABLE=1
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PROJECT_ROOT=/work/07144/yw23374/vista/context-graph
cd "$PROJECT_ROOT"

export HF_HOME=/work/07144/yw23374/vista/hf_cache
export FLASHINFER_WORKSPACE_BASE=/tmp
# Diagnostic NCCL debug + longer timeout to see where collective hangs
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2048
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace_$SLURMD_NODENAME_$$_
export TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/nccl_trace_pipe_$SLURMD_NODENAME_$$
export NCCL_TIMEOUT_MS=1200000

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
# WANDB_API_KEY from .wandb_env
# OPENAI_API_KEY not required for ALFWorld (env reward, no LLM judge)

# ── Node info ──
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

echo "════════════════════════════════════════════════════════════════"
echo "  SMOKE TEST: Qwen3-30B-A3B-Thinking-2507 on ALFWorld"
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

# ── Generate ALFWorld data (small) ──
echo "--- Generating ALFWorld parquet ---"
python scripts/make_alfworld_data.py --n_train 32 --n_val 8

# ── Ray head on Node 0 ──
echo "--- Starting Ray head on $NODE0 ---"
srun --jobid=673469 --overlap --nodes=1 --ntasks=1 -p gh-dev -t 01:30:00 -w "$NODE0" bash -c "
  source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export NCCL_HOSTID=\"\${SLURMD_NODENAME:-\$(hostname -s)}\"
  export PATH=${CONDA_PREFIX}/bin:${PATH}; hash -r
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
  export HF_HOME=/work/07144/yw23374/vista/hf_cache
export FLASHINFER_WORKSPACE_BASE=/tmp
# Diagnostic NCCL debug + longer timeout to see where collective hangs
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2048
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace_$SLURMD_NODENAME_$$_
export TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/nccl_trace_pipe_$SLURMD_NODENAME_$$
export NCCL_TIMEOUT_MS=1200000

  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  ray start --head --node-ip-address=$NODE0_IP --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
" &
RAY_HEAD_PID=$!
sleep 20

# ── Ray workers on remaining nodes ──
WORKER_PIDS=()
for i in $(seq 1 $((NUM_NODES-1))); do
  WORKER_NODE=${NODELIST[$i]}
  echo "--- Starting Ray worker on $WORKER_NODE (node $i) ---"
  srun --jobid=673469 --overlap --nodes=1 --ntasks=1 -p gh-dev -t 01:30:00 -w "$WORKER_NODE" bash -c "
    source /work/07144/yw23374/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
  export NCCL_HOSTID=\"\${SLURMD_NODENAME:-\$(hostname -s)}\"
  export PATH=${CONDA_PREFIX}/bin:${PATH}; hash -r
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
    export HF_HOME=/work/07144/yw23374/vista/hf_cache
export FLASHINFER_WORKSPACE_BASE=/tmp
# Diagnostic NCCL debug + longer timeout to see where collective hangs
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2048
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/nccl_trace_$SLURMD_NODENAME_$$_
export TORCH_NCCL_DEBUG_INFO_PIPE_FILE=/tmp/nccl_trace_pipe_$SLURMD_NODENAME_$$
export NCCL_TIMEOUT_MS=1200000

  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
    ray start --address ${NODE0_IP}:6379 --num-cpus=70 --num-gpus=1 --block
  " &
  WORKER_PIDS+=($!)
  sleep 5
done
sleep 20

export RAY_ADDRESS=${NODE0_IP}:6379
echo "--- Ray cluster status ---"
ray status || echo "WARN: ray status check failed"

# ── Run 3 RL steps just to prove the pipeline works ──
echo "════════════════════════════════════════════════════════════════"
echo "  Launching FoldGRPO BF16 smoke test (3 steps, batch=8, n=4, rollout TP=8)"
echo "════════════════════════════════════════════════════════════════"

set +e
python -m scripts.train_fold \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=/work/07144/yw23374/vista/models/Qwen3-30B-A3B-Thinking-2507 \
  actor_rollout_ref.rollout.prompt_length=512 \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=True \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  data.train_files=data/alfworld_train.parquet \
  data.val_files=data/alfworld_test.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=512 \
  data.max_response_length=2048 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=4096 \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld \
  +actor_rollout_ref.rollout.plugin.max_turn=4 \
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
  +actor_rollout_ref.rollout.plugin.val_max_turn=4 \
  +actor_rollout_ref.rollout.plugin.val_response_length=2048 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=1 \
  trainer.test_freq=999 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.experiment_name=tiny_diag_30b_fsdp1 \
  trainer.logger='["console","wandb"]'

RC=$?
echo "════════════════════════════════════════════════════════════════"
if [ $RC -eq 0 ]; then
  echo "  ✓ SMOKE TEST PASSED — environment is ready for production run"
else
  echo "  ✗ SMOKE TEST FAILED (exit $RC) — fix before launching 16-node job"
fi
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── Cleanup ──
kill $RAY_HEAD_PID 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do
  kill $pid 2>/dev/null || true
done

exit $RC
