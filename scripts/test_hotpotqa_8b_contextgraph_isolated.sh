#!/bin/bash
#SBATCH -J ctxgraph-iso-hotpot-8b
#SBATCH -o ctxgraph-iso-hotpot-8b.%j.out
#SBATCH -e ctxgraph-iso-hotpot-8b.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 72
#SBATCH -t 03:00:00
#SBATCH -A ASC24078

# ContextGraph (ISOLATED) on HotpotQA with Qwen3-8B + REAL search
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

if [ -z "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY=dummy
  echo "=== OPENAI_API_KEY not set, using dummy (EM-only grading) ==="
else
  echo "=== Using OPENAI_API_KEY from environment (${OPENAI_API_KEY:0:10}...) ==="
fi

echo "=== TEST: ContextGraph ISOLATED + Qwen3-8B on HotpotQA with REAL search ==="
echo "=== Node: $(hostname), $(date) ==="

# ── Step 1: Generate HotpotQA data ──
python scripts/make_hotpotqa_data.py --n_train 500 --n_val 100

# ── Step 2: Start REAL HotpotQA search server ──
echo "=== Starting HotpotQA TF-IDF search server ==="
fuser -k 18999/tcp 2>/dev/null || true
python scripts/hotpotqa_search_server.py &
SEARCH_PID=$!
sleep 30

for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:18999/search -d '{"query":"Einstein","k":1}' -H 'Content-Type: application/json' > /dev/null 2>&1; then
    echo "Search server ready after ${i}s"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "ERROR: Search server failed to start"
    kill $SEARCH_PID 2>/dev/null || true
    exit 1
  fi
  sleep 1
done

export LOCAL_SEARCH_URL="http://127.0.0.1:18999"

# ── Step 3: Train ──
echo "=== Starting ContextGraph (ISOLATED) + Qwen3-8B on HotpotQA (5 steps) ==="

python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.rollout.prompt_length=4096 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  data.train_files=data/hotpotqa_graph_train.parquet \
  data.val_files=data/hotpotqa_graph_test.parquet \
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
  +actor_rollout_ref.rollout.plugin.workflow=search_graph \
  +actor_rollout_ref.rollout.plugin.max_turn=8 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=512 \
  +actor_rollout_ref.rollout.plugin.max_session=3 \
  +actor_rollout_ref.rollout.plugin.val_max_session=3 \
  +actor_rollout_ref.rollout.plugin.session_timeout=600 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=2048 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope,graph]' \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=8 \
  +actor_rollout_ref.rollout.plugin.val_response_length=4096 \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_training_steps=5 \
  trainer.test_freq=5 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.experiment_name=ctxgraph_isolated_hotpotqa_8b \
  trainer.logger='["console"]'

echo "=== Test finished: $(date) ==="
kill $SEARCH_PID 2>/dev/null || true
