#!/bin/bash
#SBATCH -J train-bc-8b-base-4n-24h-v3-32k
#SBATCH -o logs/train-bc-8b-base-4n-24h-v3-32k.%j.out
#SBATCH -e logs/train-bc-8b-base-4n-24h-v3-32k.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 24:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 24-hour 4-NODE training for BrowseComp-Plus vanilla GRPO ReAct baseline (8B, 32K resp).
#
# Vanilla baseline: linear chat history + search tool, no branch / fold / graph /
# consolidation. Paired with foldagent_v3 + ctxgraph_v3 32K runs to isolate the
# contribution of the branching mechanism (fold) and the graph operations (ctxgraph)
# on top of the bare RL-trained ReAct ceiling.
#
# Zero-shot ReAct = step-0 evaluation of THIS run (val_before_train=True);
# trained ReAct  = step-30 evaluation. Both feed the (Zero-shot, Step 30) row of
# the BC-Plus main results table.
#
# 24h: total_training_steps=30 (≈ 4.2 epochs over 680-question BC-Plus train,
#       matched to fold/ctxgraph 32K paper-match runs), val + ckpt every 10 steps.
# Topology: NODELIST[0] = dedicated search server (no Ray), NODELIST[1]
# = Ray head + trainer rank 0, NODELIST[2,3] = Ray workers. FSDP across 3 GPUs.
# Production training (30 steps, val every 10 steps, ckpt every 10 steps):
#   - search_server.py loads HF datasets (Tevatron/browsecomp-plus-corpus +
#     precomputed embeddings from miaolu3/browsecomp-plus)
#   - trainer hits OpenAI judge per rollout completion (gpt-5-nano default,
#     override via JUDGE_MODEL env var; cheaper smoke: JUDGE_MODEL=gpt-4o-mini)
#   - react_agent + search_base workflow (no branch, no graph, no consol) —
#     the bare RL-trained ReAct ceiling on BC-Plus
#
# Pivot to BrowseComp-Plus because:
#   (1) MuSiQue 4B-Instruct ctxgraph showed 74% stub graphs (essentially fold)
#   (2) ALFWorld Instruct cold-starts at 0% even at 8B (Thinking variant needed)
#   (3) BC is upstream FoldAgent paper's actual benchmark — direct comparison
#   (4) BC is entity-heavy + long-horizon (200 turn cap) — exactly the regime
#       where graph compression should beat fold's branch-summary scheme
#   (5) 8B base BC ~5-15% pass@1 → plenty of RL headroom, no cold-start
#
# Pre-flight (one-time, login node):
#   gdown 'https://drive.google.com/uc?id=1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM' -O bc.zip
#   unzip bc.zip -d data/   # → data/bc_train.parquet, data/bc_test.parquet
#   hf download Qwen/Qwen3-Embedding-8B
#   hf download Tevatron/browsecomp-plus-corpus --repo-type=dataset
#   hf download miaolu3/browsecomp-plus --repo-type=dataset
#   echo 'export OPENAI_API_KEY=sk-...' > $WORK/.openai_env && chmod 600 $WORK/.openai_env
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

# ── OpenAI judge (REQUIRED for BrowseComp — no LLM judge = no reward signal) ──
if [ -n "${WORK:-}" ] && [ -f "$WORK/.openai_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.openai_env"
fi
REQUIRE_OPENAI_JUDGE=${REQUIRE_OPENAI_JUDGE:-1}
if [ "$REQUIRE_OPENAI_JUDGE" = 1 ] && { [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "dummy" ]; }; then
  echo "ERROR: OPENAI_API_KEY not set. BC training requires the OpenAI judge."
  echo "       echo 'export OPENAI_API_KEY=sk-...' > \$WORK/.openai_env && chmod 600 \$WORK/.openai_env"
  exit 1
fi
export OPENAI_API_KEY=${OPENAI_API_KEY:-dummy}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano}  # match upstream paper; override JUDGE_MODEL=gpt-4o-mini for cheaper runs

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
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}
EXTERNAL_SEARCH_URL=${EXTERNAL_SEARCH_URL:-}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── Node info ──
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}

if [ "$NUM_NODES" -ne "$EXPECTED_NUM_NODES" ]; then
  echo "Expected $EXPECTED_NUM_NODES nodes, got $NUM_NODES"
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
RUN_TAG=${RUN_TAG:-4n_24h_v3_32k}
TASK_LABEL=${TASK_LABEL:-BrowseComp-Plus}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"train_baseline_bc_8b_${RUN_TAG}_${TS}"}
TRAIN_DATA_FILE=${TRAIN_DATA_FILE:-data/bc_train.parquet}
VAL_DATA_FILE=${VAL_DATA_FILE:-data/bc_test.parquet}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
TRAINER_VAL_ONLY=${TRAINER_VAL_ONLY:-False}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-30}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-10}
BC_SEARCH_TIMEOUT_SECONDS=${BC_SEARCH_TIMEOUT_SECONDS:-600}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-foldgrpo}
PROMPT_LENGTH=${PROMPT_LENGTH:-8192}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-32768}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-40960}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-12}
ROLLOUT_N=${ROLLOUT_N:-8}
TRAIN_LR=${TRAIN_LR:-2e-6}
USE_KL_LOSS=${USE_KL_LOSS:-True}
ALGORITHM_KL_COEF=${ALGORITHM_KL_COEF:-0.005}
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.0005}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
SESSION_TIMEOUT=${SESSION_TIMEOUT:-600}
MAX_TURN=${MAX_TURN:-100}
MAX_SESSION=${MAX_SESSION:-10}
VAL_MAX_SESSION=${VAL_MAX_SESSION:-10}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-768}
FINAL_ANSWER_RESERVE=${FINAL_ANSWER_RESERVE:-2048}
FINAL_ANSWER_SAFETY_MARGIN=${FINAL_ANSWER_SAFETY_MARGIN:-64}
ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-True}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/$EXPERIMENT_NAME}

# Qwen3-8B advertises 40,960 positions. Longer evaluations must override both
# the actor/reference HF config and vLLM's independently loaded HF config.
LONG_CONTEXT_ARGS=()
if [ "$CONTEXT_LENGTH" -gt 40960 ]; then
  BC_YARN_FACTOR=${BC_YARN_FACTOR:-2.0}
  BC_YARN_ORIGINAL_LENGTH=${BC_YARN_ORIGINAL_LENGTH:-32768}
  LONG_CONTEXT_OVERRIDE="{max_position_embeddings:${CONTEXT_LENGTH},rope_scaling:{rope_type:yarn,factor:${BC_YARN_FACTOR},original_max_position_embeddings:${BC_YARN_ORIGINAL_LENGTH}}}"
  LONG_CONTEXT_ARGS+=(
    "+actor_rollout_ref.model.override_config=${LONG_CONTEXT_OVERRIDE}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides=${LONG_CONTEXT_OVERRIDE}"
  )
fi

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  TRAIN: Vanilla ReAct baseline on $TASK_LABEL (4 nodes [1 search + 3 trainer])"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Worker(s): ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  Embedder model: $EMBED_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Logger: ${probe_msg}"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: BrowseComp data parquets + HF datasets must exist ──
probe "checking $TASK_LABEL training artefacts"
if [[ "$TRAIN_DATA_FILE" = /* ]]; then TRAIN_PARQUET=$TRAIN_DATA_FILE; else TRAIN_PARQUET="$PROJECT_ROOT/$TRAIN_DATA_FILE"; fi
if [[ "$VAL_DATA_FILE" = /* ]]; then VAL_PARQUET=$VAL_DATA_FILE; else VAL_PARQUET="$PROJECT_ROOT/$VAL_DATA_FILE"; fi
for f in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    echo "       Prepare the configured train/validation parquet files before launching."
    exit 1
  fi
done
if [ -z "$EXTERNAL_SEARCH_URL" ]; then
  # HF corpus + embedding datasets (will use HF cache from \$HF_HOME/hub)
  CORPUS_DATASET="Tevatron/browsecomp-plus-corpus"
  CORPUS_EMBEDDING_DATASET="miaolu3/browsecomp-plus"
  CORPUS_CACHE_DIR="$HF_HOME/hub/datasets--${CORPUS_DATASET//\//--}"
  EMBED_CACHE_DIR_DS="$HF_HOME/hub/datasets--${CORPUS_EMBEDDING_DATASET//\//--}"
  if [ ! -d "$CORPUS_CACHE_DIR" ]; then
    echo "ERROR: corpus dataset not cached at $CORPUS_CACHE_DIR"
    echo "       login node: hf download $CORPUS_DATASET --repo-type=dataset"
    exit 1
  fi
  if [ ! -d "$EMBED_CACHE_DIR_DS" ]; then
    echo "ERROR: embedding dataset not cached at $EMBED_CACHE_DIR_DS"
    echo "       login node: hf download $CORPUS_EMBEDDING_DATASET --repo-type=dataset"
    exit 1
  fi
  probe "BC parquets + HF datasets ok"
else
  probe "configured train/validation parquets ok; external search=$EXTERNAL_SEARCH_URL"
fi

RESUME_ARGS=()
if [ -n "${RESUME_CHECKPOINT_PATH:-}" ]; then
  RESUME_PATH=$RESUME_CHECKPOINT_PATH
  if [ ! -d "$RESUME_PATH" ]; then
    echo "ERROR: checkpoint directory missing: $RESUME_PATH"
    exit 1
  fi
  RESUME_ARGS=(trainer.resume_mode=resume_path "+trainer.resume_from_path=$RESUME_PATH")
  probe "will load exact checkpoint $RESUME_PATH"
elif [ -n "${RESUME_CHECKPOINT_ROOT:-}" ]; then
  LATEST_FILE="$RESUME_CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
  if [ ! -s "$LATEST_FILE" ]; then
    echo "ERROR: missing latest checkpoint marker: $LATEST_FILE"
    exit 1
  fi
  LATEST_STEP=$(tr -d '[:space:]' < "$LATEST_FILE")
  RESUME_PATH="$RESUME_CHECKPOINT_ROOT/$LATEST_STEP"
  if [ ! -d "$RESUME_PATH" ]; then
    echo "ERROR: checkpoint directory missing: $RESUME_PATH"
    exit 1
  fi
  RESUME_ARGS=(trainer.resume_mode=resume_path "+trainer.resume_from_path=$RESUME_PATH")
  probe "will resume checkpoint $RESUME_PATH"
fi

# ── Pre-flight: 8B + embedder weights must be present (offline) ──
probe "checking model caches"
TRAINER_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
if [ ! -d "$TRAINER_CACHE_DIR" ]; then
  echo "ERROR: $MODEL_PATH not found at $TRAINER_CACHE_DIR"
  echo "       From a login node, run: hf download $MODEL_PATH"
  exit 1
fi
probe "trainer cache: $TRAINER_CACHE_DIR"
if [ -z "$EXTERNAL_SEARCH_URL" ]; then
  EMBED_CACHE_DIR="$HF_HUB_CACHE/models--${EMBED_MODEL//\//--}"
  if [ ! -d "$EMBED_CACHE_DIR" ]; then
    echo "ERROR: $EMBED_MODEL not found at $EMBED_CACHE_DIR"
    echo "       From a login node, run: hf download $EMBED_MODEL"
    exit 1
  fi
  probe "embedder cache: $EMBED_CACHE_DIR"
fi

# ── Topology: NODE0 = dedicated search server (no Ray), NODE1 = Ray head, NODE2-3 = workers ──
SEARCH_NODE=${NODELIST[0]}
SEARCH_NODE_IP=$(getent hosts "$SEARCH_NODE" | awk '{print $1}')
TRAINER_HEAD_NODE=${NODELIST[1]}
TRAINER_HEAD_IP=$(getent hosts "$TRAINER_HEAD_NODE" | awk '{print $1}')
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$SEARCH_NODE,$SEARCH_NODE_IP,$TRAINER_HEAD_NODE,$TRAINER_HEAD_IP"
export no_proxy="$NO_PROXY"
echo "  Dedicated search node: $SEARCH_NODE ($SEARCH_NODE_IP)"
echo "  Trainer Ray head:      $TRAINER_HEAD_NODE ($TRAINER_HEAD_IP)"
echo "  Trainer workers:       ${NODELIST[@]:2}"

# ── Start envs/search_server.py on dedicated SEARCH_NODE ──
SEARCH_PID=""
if [ -n "$EXTERNAL_SEARCH_URL" ]; then
  export LOCAL_SEARCH_URL="$EXTERNAL_SEARCH_URL"
  probe "using wrapper-managed search server at $LOCAL_SEARCH_URL"
else
  probe "starting envs/search_server.py with $EMBED_MODEL on dedicated $SEARCH_NODE:18999"
  srun --overlap --nodes=1 --ntasks=1 -w "$SEARCH_NODE" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  cd $PROJECT_ROOT
  export PYTHONPATH=$PROJECT_ROOT:\${PYTHONPATH:-}
  export HF_HOME=$HF_HOME
  export HF_HUB_CACHE=$HF_HUB_CACHE
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE  # search server needs HF Hub for BC load_dataset (cache still preferred)
  export NUM_GPUS=1
  export MAX_BATCH_SIZE=128
  unset LOCAL_CORPUS_PARQUET LOCAL_EMBEDDINGS_PKL  # use HF dataset mode for BC
  exec python -u envs/search_server.py \
    --model $EMBED_MODEL \
    --port 18999 \
    --corpus Tevatron/browsecomp-plus-corpus \
    --corpus-embedding-dataset miaolu3/browsecomp-plus
  " >/tmp/hp_server_$$.log 2>&1 &
  SEARCH_PID=$!

  probe "waiting for search server /health (up to ${BC_SEARCH_TIMEOUT_SECONDS}s)"
  HEALTH_OK=0
  for _ in $(seq 1 "$BC_SEARCH_TIMEOUT_SECONDS"); do
    if curl --noproxy '*' -fsS "http://${SEARCH_NODE_IP}:18999/health" >/dev/null 2>&1; then
      HEALTH_OK=1
      break
    fi
    if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
      echo "ERROR: search server exited before becoming healthy. Last 80 lines:"
      tail -80 /tmp/hp_server_$$.log || true
      exit 1
    fi
    sleep 1
  done
  if [ "$HEALTH_OK" != "1" ]; then
    echo "ERROR: search server did not become healthy within ${BC_SEARCH_TIMEOUT_SECONDS}s. Last 80 lines:"
    tail -80 /tmp/hp_server_$$.log || true
    kill "$SEARCH_PID" 2>/dev/null || true
    exit 1
  fi
  export LOCAL_SEARCH_URL="http://${SEARCH_NODE_IP}:18999"
fi

probe "waiting for search server /search probe (up to ${BC_SEARCH_TIMEOUT_SECONDS}s)"
SEARCH_OK=0
for _ in $(seq 1 "$BC_SEARCH_TIMEOUT_SECONDS"); do
  if curl --noproxy '*' -fsS -X POST -H 'Content-Type: application/json' \
      -d '{"query":"Eiffel Tower","k":1}' \
      "$LOCAL_SEARCH_URL/search" >/dev/null 2>&1; then
    SEARCH_OK=1
    break
  fi
  if [ -n "$SEARCH_PID" ] && ! kill -0 "$SEARCH_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done
if [ "$SEARCH_OK" != "1" ]; then
  echo "ERROR: search server not reachable at $LOCAL_SEARCH_URL"
  if [ -n "$SEARCH_PID" ]; then
    tail -80 /tmp/hp_server_$$.log || true
    kill "$SEARCH_PID" 2>/dev/null || true
  fi
  exit 1
fi
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

# ── Ray head on TRAINER_HEAD_NODE (NODELIST[1]); search node is excluded from Ray ──
probe "starting Ray head on $TRAINER_HEAD_NODE (NODELIST[1])"
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" bash -c '
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
  ray start --head --node-ip-address='"$TRAINER_HEAD_IP"' --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
' &
RAY_HEAD_PID=$!
sleep 20
probe "Ray head sleep done; launching $((NUM_NODES - 2)) trainer workers (skip search node)"

# ── Ray workers on NODELIST[1..N-1] ──
WORKER_PIDS=()
for i in $(seq 2 $((NUM_NODES - 1))); do  # skip NODELIST[0]=search, [1]=head
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
    ray start --address='"${TRAINER_HEAD_IP}:6379"' --num-cpus=70 --num-gpus=1 --block
  ' &
  WORKER_PIDS+=("$!")
  sleep 5
done
sleep 20
probe "all $((NUM_NODES - 2)) trainer workers launched, cluster settling"

cleanup() {
  if [ -n "$SEARCH_PID" ]; then
    kill "$SEARCH_PID" 2>/dev/null || true
  fi
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

export RAY_ADDRESS=${TRAINER_HEAD_IP}:6379
probe "querying ray status"
ray status || echo "WARN: ray status check failed"

echo "=============================================================="
echo "  Launching Vanilla ReAct GRPO baseline on $TASK_LABEL"
echo "  workflow=search_base, no branch/graph/consolidation, process_reward=[flat]"
echo "  vLLM gpu_memory_utilization=0.6 + FSDP CPU offload"
echo "  val_before_train=True (step 0 = zero-shot ReAct), val every 10 steps, save every 10 steps"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" --chdir="$PROJECT_ROOT" \
  --export=ALL,LOCAL_SEARCH_URL="$LOCAL_SEARCH_URL",OPENAI_API_KEY="$OPENAI_API_KEY",JUDGE_MODEL="$JUDGE_MODEL" \
  python -m scripts.train_baseline \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  algorithm.kl_ctrl.kl_coef="$ALGORITHM_KL_COEF" \
  actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  "${LONG_CONTEXT_ARGS[@]}" \
  actor_rollout_ref.rollout.prompt_length="$PROMPT_LENGTH" \
  actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$CONTEXT_LENGTH" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.optim.lr="$TRAIN_LR" \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
  actor_rollout_ref.actor.grad_clip=0.5 \
  actor_rollout_ref.actor.kl_loss_coef="$ACTOR_KL_LOSS_COEF" \
  data.train_files="$TRAIN_DATA_FILE" \
  data.val_files="$VAL_DATA_FILE" \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$PROMPT_LENGTH" \
  data.max_response_length="$RESPONSE_LENGTH" \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$CONTEXT_LENGTH" \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu="$CONTEXT_LENGTH" \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking="$ENTROPY_FROM_LOGITS_WITH_CHUNKING" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=search_base \
  +actor_rollout_ref.rollout.plugin.max_turn="$MAX_TURN" \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens="$TURN_MAX_NEW_TOKENS" \
  +actor_rollout_ref.rollout.plugin.final_answer_reserve="$FINAL_ANSWER_RESERVE" \
  +actor_rollout_ref.rollout.plugin.final_answer_safety_margin="$FINAL_ANSWER_SAFETY_MARGIN" \
  +actor_rollout_ref.rollout.plugin.max_session="$MAX_SESSION" \
  +actor_rollout_ref.rollout.plugin.val_max_session="$VAL_MAX_SESSION" \
  +actor_rollout_ref.rollout.plugin.session_timeout="$SESSION_TIMEOUT" \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len="$RESPONSE_LENGTH" \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat]' \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.2 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.002 \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_max_turn="$MAX_TURN" \
  +actor_rollout_ref.rollout.plugin.val_response_length="$RESPONSE_LENGTH" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.val_only="$TRAINER_VAL_ONLY" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=$((NUM_NODES - 1)) \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.default_local_dir="$CHECKPOINT_ROOT" \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER" \
  "${RESUME_ARGS[@]}"
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
