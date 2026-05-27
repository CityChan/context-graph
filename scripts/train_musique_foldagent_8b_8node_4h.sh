#!/bin/bash
#SBATCH -J train-mu-8b-fa-8n
#SBATCH -o logs/train-mu-8b-fa-8n.%j.out
#SBATCH -e logs/train-mu-8b-fa-8n.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 4-hour 8-NODE FoldAgent baseline training on MuSiQue at 8B (30-step probe, 16K resp).
# Plan A counterpart of train_musique_ctxgraph_8b_8node_4h.sh.
# Goal: test whether fold can learn at 2x response budget (16K) when it failed
# to learn at 8K. If fold task_reward starts climbing at 16K, validates the
# token-efficiency framing: graph wins at small budget, fold needs more room.
# Pairs with train_musique_ctxgraph_8b_4node_4h.sh — same backbone, same
# retrieval, same data domain, only the agent architecture differs:
#   default_agent_loop=fold_agent       (vs context_graph_isolated_agent)
#   plugin.workflow=search_branch       (vs search_graph)
#   plugin.process_reward='[flat,scope]' (vs '[flat,scope,graph]')
#   data files = data/musique_{train,test}.parquet
#                                       (vs musique_graph_{train,test})
#
# MuSiQue (4-hop QA) is selected because it's where graph data structure
# SHOULD differentiate: chain-of-evidence reasoning, many entities cross
# multi-turn, fold's flat history loses earlier-hop entities by step 4.
#
# Pre-flight (one-time):
#   (login) python scripts/make_hotpotqa_data.py
#           — note this script must produce BOTH hotpotqa_{train,test}
#             (fold) AND hotpotqa_graph_{train,test} (ctxgraph).
#   (login) python scripts/build_unified_wiki_corpus.py
#   (sbatch) scripts/build_unified_wiki_index.sh
#   (login) hf download Qwen/Qwen3-8B
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
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── Node info ──
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

if [ "$NUM_NODES" -ne 8 ]; then
  echo "Expected 8 nodes (set #SBATCH -N 8 or use idev -N 8), got $NUM_NODES"
  exit 1
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  probe_msg="wandb enabled (key length=${#WANDB_API_KEY})"
else
  TRAINER_LOGGER='["console"]'
  probe_msg="WARNING: no WANDB_API_KEY in env — training will only log to console"
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="train_foldagent_musique_8b_8n_4h_${TS}"

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  TRAIN: FoldAgent on MuSiQue (8B, 8 nodes, 30 steps, 4h, 16K-resp)"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Worker(s): ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  Embedder model: $EMBED_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Logger: ${probe_msg}"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: MuSiQue fold artefacts must already exist ──
probe "checking MuSiQue fold artefacts"
TRAIN_PARQUET="$PROJECT_ROOT/data/musique_train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data/musique_test.parquet"
CORPUS_PARQUET="$PROJECT_ROOT/data/wiki_corpus.parquet"
EMBED_PKL="$PROJECT_ROOT/data/wiki_corpus_embeddings.pkl"
for f in "$TRAIN_PARQUET" "$VAL_PARQUET" "$CORPUS_PARQUET" "$EMBED_PKL"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    echo "       Fold uses data/musique_{train,test}.parquet (no _graph_ infix)."
    echo "       Re-run scripts/make_musique_data.py if either is missing."
    exit 1
  fi
done
probe "MuSiQue fold artefacts ok"

# ── Pre-flight: 8B + embedder weights must be present (offline) ──
probe "checking model caches"
TRAINER_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
EMBED_CACHE_DIR="$HF_HUB_CACHE/models--${EMBED_MODEL//\//--}"
if [ ! -d "$TRAINER_CACHE_DIR" ]; then
  echo "ERROR: $MODEL_PATH not found at $TRAINER_CACHE_DIR"
  echo "       From a login node, run: hf download $MODEL_PATH"
  exit 1
fi
if [ ! -d "$EMBED_CACHE_DIR" ]; then
  echo "ERROR: $EMBED_MODEL not found at $EMBED_CACHE_DIR"
  echo "       From a login node, run: hf download $EMBED_MODEL"
  exit 1
fi
probe "trainer cache: $TRAINER_CACHE_DIR"
probe "embedder cache: $EMBED_CACHE_DIR"

# ── Start envs/search_server.py on NODE0, in background ──
probe "starting envs/search_server.py with $EMBED_MODEL on $NODE0:18999"
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  cd $PROJECT_ROOT
  export PYTHONPATH=$PROJECT_ROOT:\${PYTHONPATH:-}
  export HF_HOME=$HF_HOME
  export HF_HUB_CACHE=$HF_HUB_CACHE
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

# ── Stale Ray cleanup on both nodes ──
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
probe "python sanity imports"
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
probe "sanity imports done"

# ── Ray head on NODE0 ──
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
  export HF_HUB_CACHE='"$HF_HUB_CACHE"'
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

# ── Ray workers on NODELIST[1..N-1] ──
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
    export HF_HUB_CACHE='"$HF_HUB_CACHE"'
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
probe "all $((NUM_NODES - 1)) Ray workers launched, cluster settling"

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
echo "  Launching FoldAgent FoldGRPO training (8 nodes, 30 steps, 16K resp, MuSiQue)"
echo "  vLLM gpu_memory_utilization=0.55 + FSDP CPU offload"
echo "  val_before_train=True (step-0 anchor), save_freq=15"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$NODE0" --chdir="$PROJECT_ROOT" \
  --export=ALL,LOCAL_SEARCH_URL="$LOCAL_SEARCH_URL" \
  python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.response_length=16384 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=18432 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=8 \
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
  actor_rollout_ref.actor.grad_clip=0.5 \
  actor_rollout_ref.actor.kl_loss_coef=0.0005 \
  data.train_files=data/musique_train.parquet \
  data.val_files=data/musique_test.parquet \
  data.train_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=16384 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=18432 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=search_branch \
  +actor_rollout_ref.rollout.plugin.max_turn=20 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=768 \
  +actor_rollout_ref.rollout.plugin.max_session=3 \
  +actor_rollout_ref.rollout.plugin.val_max_session=3 \
  +actor_rollout_ref.rollout.plugin.session_timeout=600 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=4096 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope]' \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=20 \
  +actor_rollout_ref.rollout.plugin.val_response_length=16384 \
  trainer.val_before_train=True \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=30 \
  trainer.test_freq=999 \
  trainer.save_freq=15 \
  trainer.default_local_dir=${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/$EXPERIMENT_NAME \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER"
RC=$?
set -e

echo "=============================================================="
if [ $RC -eq 0 ]; then
  echo "  TRAIN RUN COMPLETED (exit 0)"
else
  echo "  TRAIN RUN FAILED (exit $RC) — check above for first error"
fi
echo "  Finished: $(date)"
echo "=============================================================="

exit $RC
