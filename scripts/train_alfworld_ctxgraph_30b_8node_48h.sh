#!/bin/bash
#SBATCH -J cg-alf-30b-8n
#SBATCH -o cg-alf-30b-8n.%j.out
#SBATCH -e cg-alf-30b-8n.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# Production (8-node): ContextGraph isolated / Qwen3-30B-A3B-Thinking-2507
#                       / ALFWorld @hard
#   8 nodes × 1 GH200 = 8 GPUs (FSDP)  |  48 hours  |  AST24021
#
# Differences vs 16-node yating script:
#   * #SBATCH -N 8 (half the nodes; expect ~1.5-2x slower per step)
#   * total_training_steps=300 (vs 500 on 16n; 8n with batch=32 won't
#     finish 500 in 48h)
#   * save_freq=50 (more checkpoints since steps cut to 300)
#   * data gen uses --hard (yating's script omitted --hard so it would
#     overwrite hard data with real every launch — fixed here)
#
# Pairs with train_alfworld_fold_30b_8node_48h.sh — same hyperparams,
# same data, only the agent loop + workflow + reward differ:
#   agent loop  : context_graph_isolated_agent (per-branch subgraph)
#   workflow    : alfworld_graph (graph tools enabled in prompt)
#   data        : data/alfworld_graph_{train,test}.parquet
#   reward      : flat + scope + graph (outcome-only graph reward)
#   extras      : lambda_compact 0.1, lambda_cost 0.005
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

# Memory polish (ported from yating's gh-dev mempolish smoke):
#   - expandable_segments cuts CUDA allocator fragmentation across long rollouts
#   - Ray host-mem kill disabled so workers don't get reaped under transient pressure
#   - per-PID Triton/vLLM caches avoid shared-FS contention across the 8 nodes
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp

# Pin HF cache offline. Hypothesis for job 704096 tokenizer_config.json JSONDecodeError:
# concurrent rank refetches racing on the same file. Offline mode reads cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_FILE_LOCKING=1

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"

export HF_HOME=/work/09281/chc_1996/vista/cache
# Vista cache uses legacy layout ($HF_HOME/models--XXX), not $HF_HOME/hub/.
# Override HF_HUB_CACHE so new `hf`/transformers tooling reads the existing
# 60GB of Apr 29 weights instead of looking at an empty hub/ subdir.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export NCCL_P2P_LEVEL=NVL
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY in your shell before running this script}"

# ── Node info ──
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

if [ "$NUM_NODES" -ne 8 ]; then
  echo "Expected 8 nodes (set #SBATCH -N 8), got $NUM_NODES"
  exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="ctxgraph_alfworld_hard_30b_8n_${TS}"

echo "════════════════════════════════════════════════════════════════"
echo "  PRODUCTION (8n): ContextGraph (isolated) / Qwen3-30B-A3B / ALFWorld @hard"
echo "  Nodes: $NUM_NODES   Head: $NODE0 ($NODE0_IP)"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Started: $(date)"
echo "════════════════════════════════════════════════════════════════"
for i in $(seq 0 $((NUM_NODES-1))); do
  echo "    Node $i: ${NODELIST[$i]}"
done

# ── Generate ALFWorld data (creates both alfworld_*.parquet and alfworld_graph_*.parquet) ──
# CRITICAL: --hard for MemexRL-style mode (no admissible commands in obs)
echo "--- Generating ALFWorld parquet (hard mode) ---"
python scripts/make_alfworld_data.py --hard --n_train 1000 --n_val 100

# Sanity-check the parquet has @hard ability
python -c "
import pandas as pd
df = pd.read_parquet('data/alfworld_graph_train.parquet')
assert df['ability'].iloc[0] == 'ALFWorld@hard', f'expected hard, got {df[\"ability\"].iloc[0]}'
print(f'OK: graph_train.parquet ability={df[\"ability\"].iloc[0]} rows={len(df)}')
"

# ── Ray head ──
echo "--- Starting Ray head on $NODE0 ---"
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
  export HF_HOME=/work/09281/chc_1996/vista/cache
  export HF_HUB_CACHE=/work/09281/chc_1996/vista/cache
  export ALFWORLD_DATA=$ALFWORLD_DATA
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_FILE_LOCKING=1
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    export ALFWORLD_DATA=$ALFWORLD_DATA
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_FILE_LOCKING=1
    export FLASHINFER_WORKSPACE_BASE=/tmp
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
echo "  Launching FoldGRPO training: ContextGraph isolated"
echo "  Target 300 steps (~48h on 8 nodes)"
echo "════════════════════════════════════════════════════════════════"

python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Thinking-2507 \
  actor_rollout_ref.rollout.prompt_length=8192 \
  actor_rollout_ref.rollout.response_length=32768 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  data.train_files=data/alfworld_graph_train.parquet \
  data.val_files=data/alfworld_graph_test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=8192 \
  data.max_response_length=32768 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=40960 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld_graph \
  +actor_rollout_ref.rollout.plugin.max_turn=50 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.max_session=10 \
  +actor_rollout_ref.rollout.plugin.val_max_session=10 \
  +actor_rollout_ref.rollout.plugin.session_timeout=1800 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=8192 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope,graph]' \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
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
  trainer.total_training_steps=300 \
  trainer.test_freq=25 \
  trainer.save_freq=50 \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger='["console","wandb"]'

echo "════════════════════════════════════════════════════════════════"
echo "  Training finished: $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── Cleanup ──
kill $RAY_HEAD_PID 2>/dev/null || true
for pid in "${WORKER_PIDS[@]}"; do
  kill $pid 2>/dev/null || true
done
