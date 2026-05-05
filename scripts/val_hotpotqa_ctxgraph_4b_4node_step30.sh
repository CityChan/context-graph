#!/bin/bash
#SBATCH -J val-cg-hp-4b
#SBATCH -o val-cg-hp-4b.%j.out
#SBATCH -e val-cg-hp-4b.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 03:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# Validation-only run: load ContextGraph step 30 ckpt and run val once,
# producing clean val/reward + val/avg_score numbers that the original
# 4h training run never wrote to wandb (it crashed during final cleanup
# after ckpt save was complete).
#
# Default points at:
#   checkpoints/context-graph/ctxgraph_hotpotqa_4b_4n_p2048_r8192_4h_20260504_094954/global_step_30
# Override with CKPT_PATH=/abs/path/to/global_step_NN env var.
#
# Mechanism: trainer.val_only=True returns after the initial validation
# (see verl/trainer/ppo/ray_trainer.py line 1063). resume_mode=resume_path
# tells the FSDP worker to load model + optim shards from CKPT_PATH/actor.
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
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Thinking-2507}
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── Node info ──
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

if [ "$NUM_NODES" -ne 4 ]; then
  echo "Expected 4 nodes (set #SBATCH -N 4), got $NUM_NODES"
  exit 1
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
else
  TRAINER_LOGGER='["console"]'
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="val_ctxgraph_hotpotqa_4b_4n_step30_${TS}"

# ── Checkpoint to evaluate (override via CKPT_PATH env var) ──
DEFAULT_CKPT="$PROJECT_ROOT/checkpoints/context-graph/ctxgraph_hotpotqa_4b_4n_p2048_r8192_4h_20260504_094954/global_step_30"
CKPT_PATH=${CKPT_PATH:-$DEFAULT_CKPT}
if [ ! -d "$CKPT_PATH/actor" ]; then
  echo "ERROR: ckpt not found at $CKPT_PATH/actor"
  echo "  Override with CKPT_PATH=/abs/path/to/global_step_NN"
  exit 1
fi
echo "  Loading ckpt: $CKPT_PATH"

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  ContextGraph (isolated) on HotpotQA (4B, 4 nodes)"
echo "  Job: $SLURM_JOB_ID   Head: $NODE0 ($NODE0_IP)"
echo "  Trainer model:  $MODEL_PATH"
echo "  Embedder model: $EMBED_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: HotpotQA artefacts must already exist ──
probe "checking HotpotQA artefacts"
TRAIN_PARQUET="$PROJECT_ROOT/data/hotpotqa_graph_train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data/hotpotqa_graph_test.parquet"
CORPUS_PARQUET="$PROJECT_ROOT/data/wiki_corpus.parquet"
EMBED_PKL="$PROJECT_ROOT/data/wiki_corpus_embeddings.pkl"
for f in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CORPUS_PARQUET" "$EMBED_PKL"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    echo
    echo "Stage-2 prep, in order:"
    echo "  (login node)   python scripts/make_hotpotqa_data.py"
    echo "  (login node)   python scripts/build_unified_wiki_corpus.py"
    echo "  (compute node) sbatch scripts/build_unified_wiki_index.sh"
    exit 1
  fi
done
probe "HotpotQA artefacts ok"

# ── Pre-download trainer model on head node (single process, avoids NFS race) ──
probe "checking HF model cache for $MODEL_PATH"
TRAINER_CACHE_DIR="$HF_HOME/hub/models--$(echo "$MODEL_PATH" | sed 's|/|--|g')/snapshots"
if [ -d "$TRAINER_CACHE_DIR" ] && [ -n "$(find "$TRAINER_CACHE_DIR" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
  probe "trainer cache hit"
else
  probe "trainer cache miss; downloading $MODEL_PATH on head node"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    huggingface-cli download "$MODEL_PATH" --cache-dir "$HF_HOME" || {
      echo "Trainer model download failed."
      exit 1
    }
fi

# ── Pre-download embedder model if missing (small, fast) ──
EMBED_SNAPSHOT_DIR="$HF_HOME/hub/models--$(echo "$EMBED_MODEL" | sed 's|/|--|g')/snapshots"
if [ ! -d "$EMBED_SNAPSHOT_DIR" ] || [ -z "$(ls -A "$EMBED_SNAPSHOT_DIR" 2>/dev/null)" ]; then
  probe "embedder cache miss; downloading $EMBED_MODEL on head node"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
    huggingface-cli download "$EMBED_MODEL" --cache-dir "$HF_HOME" || {
      echo "Embedder model download failed."
      exit 1
    }
fi

# ── Start envs/search_server.py on NODE0, in background ──
# It will spawn 1 worker on NODE0's GPU 0; the trainer's vLLM is capped via
# gpu_memory_utilization=0.5 below to leave room for the embedder + corpus.
probe "starting envs/search_server.py with $EMBED_MODEL on $NODE0:18999"
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  cd $PROJECT_ROOT
  export PYTHONPATH=$PROJECT_ROOT:\${PYTHONPATH:-}
  export HF_HOME=$HF_HOME
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export NUM_GPUS=1
  export MAX_BATCH_SIZE=64
  export LOCAL_CORPUS_PARQUET=$CORPUS_PARQUET
  export LOCAL_EMBEDDINGS_PKL=$EMBED_PKL
  exec python -u envs/search_server.py \
    --model $EMBED_MODEL \
    --port 18999 \
    --local-corpus $CORPUS_PARQUET \
    --local-embeddings $EMBED_PKL
" >/tmp/hp_server_$$.log 2>&1 &
SEARCH_PID=$!

# Boot can take 1-3 min: model load on remote node + embeddings move-to-GPU
probe "waiting for search server /health (up to 240s)"
for i in $(seq 1 120); do
  if curl -fsS "http://${NODE0_IP}:18999/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl -fsS -X POST -H 'Content-Type: application/json' \
        -d '{"query":"Eiffel Tower","k":1}' \
        "http://${NODE0_IP}:18999/search" >/dev/null; then
  echo "ERROR: search server not reachable at ${NODE0_IP}:18999. Last 80 lines:"
  tail -80 /tmp/hp_server_$$.log || true
  kill "$SEARCH_PID" 2>/dev/null || true
  exit 1
fi
export LOCAL_SEARCH_URL="http://${NODE0_IP}:18999"
probe "search server up at $LOCAL_SEARCH_URL"

# ── Stale Ray cleanup on all allocated nodes ──
probe "ray stop sweep across $NUM_NODES nodes"
for node in "${NODELIST[@]}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -c '
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    ray stop -f >/dev/null 2>&1 || true
  ' || true
done
sleep 5
probe "ray stop sweep done"

# ── Sanity ──
probe "python sanity imports (cold torch import can take ~60s)"
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
probe "sanity imports done"

# ── Ray head ──
probe "starting Ray head on $NODE0"
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" bash -c '
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  hash -r
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
  export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
  export HF_HOME='"$HF_HOME"'
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export LOCAL_SEARCH_URL='"$LOCAL_SEARCH_URL"'
  ray start --head --node-ip-address='"$NODE0_IP"' --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
' &
RAY_HEAD_PID=$!
sleep 20
probe "Ray head sleep done; launching $((NUM_NODES - 1)) workers"

# ── Ray workers ──
WORKER_PIDS=()
for i in $(seq 1 $((NUM_NODES - 1))); do
  WORKER_NODE=${NODELIST[$i]}
  srun --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c '
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    hash -r
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
    export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH}
    export HF_HOME='"$HF_HOME"'
    export FLASHINFER_WORKSPACE_BASE=/tmp
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export LOCAL_SEARCH_URL='"$LOCAL_SEARCH_URL"'
    ray start --address='"${NODE0_IP}:6379"' --num-cpus=70 --num-gpus=1 --block
  ' &
  WORKER_PIDS+=("$!")
  sleep 5
done
sleep 20
probe "all Ray workers launched, cluster settling"

cleanup() {
  kill "$SEARCH_PID" 2>/dev/null || true
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

export RAY_ADDRESS=${NODE0_IP}:6379
probe "querying ray status"
ray status || echo "WARN: ray status check failed"

echo "=============================================================="
echo "  Launching ContextGraph VAL-ONLY (4 nodes, ckpt=$CKPT_PATH)"
echo "  vLLM gpu_memory_utilization=0.5 (NODE0 shares its GPU with the embedder)"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min before first wandb log)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" --chdir="$PROJECT_ROOT" \
  --export=ALL,LOCAL_SEARCH_URL="$LOCAL_SEARCH_URL" \
  python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  data.train_files=data/hotpotqa_graph_train.parquet \
  data.val_files=data/hotpotqa_graph_test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=8192 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=10240 \
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
  +actor_rollout_ref.rollout.plugin.val_response_length=8192 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.resume_mode=resume_path \
  +trainer.resume_from_path="$CKPT_PATH" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=1 \
  trainer.test_freq=999 \
  trainer.save_freq=-1 \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER"
RC=$?
set -e

echo "=============================================================="
if [ $RC -eq 0 ]; then
  echo "  VAL-ONLY COMPLETED (exit 0) — see wandb run $EXPERIMENT_NAME for val/reward"
else
  echo "  VAL-ONLY FAILED (exit $RC)"
fi
echo "  Finished: $(date)"
echo "=============================================================="

exit $RC
