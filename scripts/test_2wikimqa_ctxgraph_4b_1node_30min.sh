#!/bin/bash
#SBATCH -J test-2w-4b-cg-1n
#SBATCH -o test-2w-4b-cg-1n.%j.out
#SBATCH -e test-2w-4b-cg-1n.%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 00:30:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 30-minute single-node smoke for ContextGraph on 2WikiMQA, runnable on
# either sbatch or directly inside an idev session:
#   idev -p gh-dev -N 1 -t 00:30:00 -A AST24021
#   bash scripts/test_2wikimqa_ctxgraph_4b_1node_30min.sh
#
# Uses the 4096/6144/0.5 morning baseline that was confirmed working
# this morning (response_length=4096, ppo_max_token_len=6144,
# vllm gpu_memory_utilization=0.5, batch=16). Trajectories cap at
# ~10 turns (4096 / 384) which is fine for a smoke; the production
# 4-node script bumps response_length=8192 for full 20-turn coverage.
#
# Trims for budget:
#   total_training_steps=2 (catches step-1 -> step-2 transition)
#   val_before_train=False, save_freq=-1, test_freq=999
#
# Pre-flight (one-time, before this script):
#   (login)   python scripts/make_2wikimqa_data.py
#   (login)   python scripts/build_unified_wiki_corpus.py
#   (compute) sbatch scripts/build_unified_wiki_index.sh
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Vista cache redirects (avoid NFS flock) ──
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

# ── WANDB (optional) ──
if [ -n "${WORK:-}" ] && [ -f "$WORK/.wandb_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.wandb_env"
fi
export WANDB_API_KEY=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

# ── Conda + CUDA ──
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LIBRARY_PATH:-}
export CPATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/include:${CPATH:-}

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCHDYNAMO_DISABLE=1
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=NVL

# ── Project paths ──
PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
else
  TRAINER_LOGGER='["console"]'
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="test_ctxgraph_2wikimqa_4b_1n_smoke30min_${TS}"

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  SMOKE: ContextGraph on 2WikiMQA (4B, 1 node, 2 steps)"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Host: $(hostname -s)"
echo "  Trainer model:  $MODEL_PATH"
echo "  Embedder model: $EMBED_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: 2WikiMQA artefacts must already exist ──
probe "checking 2WikiMQA artefacts"
TRAIN_PARQUET="$PROJECT_ROOT/data/2wikimqa_graph_train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data/2wikimqa_graph_test.parquet"
CORPUS_PARQUET="$PROJECT_ROOT/data/wiki_corpus.parquet"
EMBED_PKL="$PROJECT_ROOT/data/wiki_corpus_embeddings.pkl"
for f in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CORPUS_PARQUET" "$EMBED_PKL"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    echo
    echo "Stage-2 prep, in order:"
    echo "  (login node)   python scripts/make_2wikimqa_data.py"
    echo "  (login node)   python scripts/build_unified_wiki_corpus.py"
    echo "  (compute node) sbatch scripts/build_unified_wiki_index.sh"
    exit 1
  fi
done
probe "2WikiMQA artefacts ok"

# ── Start the embedding-based search server in the background ──
probe "starting envs/search_server.py with $EMBED_MODEL on localhost:18999"
export LOCAL_CORPUS_PARQUET="$CORPUS_PARQUET"
export LOCAL_EMBEDDINGS_PKL="$EMBED_PKL"
export NUM_GPUS=1
export MAX_BATCH_SIZE=16
python -u envs/search_server.py \
  --model "$EMBED_MODEL" \
  --port 18999 \
  --local-corpus "$CORPUS_PARQUET" \
  --local-embeddings "$EMBED_PKL" \
  >/tmp/hp_server_$$.log 2>&1 &
SEARCH_PID=$!

probe "waiting for search server /health (up to 180s)"
for i in $(seq 1 90); do
  if curl -fsS "http://localhost:18999/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl -fsS -X POST -H 'Content-Type: application/json' \
        -d '{"query":"Eiffel Tower","k":1}' \
        http://localhost:18999/search >/dev/null; then
  echo "ERROR: search server did not come up. Last 80 lines of its log:"
  tail -80 /tmp/hp_server_$$.log || true
  kill "$SEARCH_PID" 2>/dev/null || true
  exit 1
fi
export LOCAL_SEARCH_URL="http://localhost:18999"
probe "search server up at $LOCAL_SEARCH_URL"

# ── Stale Ray cleanup on this node ──
probe "ray stop"
ray stop -f >/dev/null 2>&1 || true
sleep 3

# ── Sanity ──
probe "python sanity imports"
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
probe "sanity imports done"

# ── Ray head on this node only ──
probe "starting Ray head on localhost"
ray start --head --port=6379 --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 \
  >/tmp/ray_head_$$.log 2>&1
sleep 10
export RAY_ADDRESS="127.0.0.1:6379"
ray status || echo "WARN: ray status check failed"

cleanup() {
  kill "$SEARCH_PID" 2>/dev/null || true
  ray stop -f >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=============================================================="
echo "  Launching ContextGraph FoldGRPO smoke (1 node, 2 steps)"
echo "  vLLM gpu_memory_utilization=0.5 + FSDP CPU offload"
echo "=============================================================="
probe "launching trainer"

set +e
python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6144 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  data.train_files=data/2wikimqa_graph_train.parquet \
  data.val_files=data/2wikimqa_graph_test.parquet \
  data.train_batch_size=16 \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=6144 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=search_graph \
  +actor_rollout_ref.rollout.plugin.max_turn=20 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=384 \
  +actor_rollout_ref.rollout.plugin.max_session=3 \
  +actor_rollout_ref.rollout.plugin.val_max_session=3 \
  +actor_rollout_ref.rollout.plugin.session_timeout=300 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=2048 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope,graph]' \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
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
  trainer.total_training_steps=2 \
  trainer.test_freq=999 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER"
RC=$?
set -e

echo "=============================================================="
if [ $RC -eq 0 ]; then
  echo "  SMOKE PASSED (exit 0) — ContextGraph 1-node link verified"
else
  echo "  SMOKE FAILED (exit $RC) — check above for first error"
fi
echo "  Finished: $(date)"
echo "=============================================================="

exit $RC
