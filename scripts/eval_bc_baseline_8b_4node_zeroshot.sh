#!/bin/bash
#SBATCH -J eval-bc-8b-base-zeroshot
#SBATCH -o logs/eval-bc-8b-base-zeroshot.%j.out
#SBATCH -e logs/eval-bc-8b-base-zeroshot.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 1:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 1-hour 4-NODE ZERO-SHOT eval for BrowseComp-Plus vanilla ReAct (8B, 32K resp).
#
# Same agent / workflow / topology as train_bc_baseline_8b_4node_24h_v3_32k.sh,
# but trainer.val_only=True → run validation against bc_test.parquet (150
# questions) and exit. No RL training, no checkpoint save.
#
# Use case: get a zero-shot ReAct baseline number for the paper table without
# committing to the 24h training slot. Equivalent to step 0 of the matching
# training run, but standalone so the train slot can queue independently.
#
# Expected wall clock: model init ~3-5 min + greedy val rollout ~10-15 min
# + LLM judge serial overhead ~5 min ≈ 20-25 min total. Padded to 1h.
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
if [ -z "${OPENAI_API_KEY:-}" ] || [ "$OPENAI_API_KEY" = "dummy" ]; then
  echo "ERROR: OPENAI_API_KEY not set. BC training requires the OpenAI judge."
  echo "       echo 'export OPENAI_API_KEY=sk-...' > \$WORK/.openai_env && chmod 600 \$WORK/.openai_env"
  exit 1
fi
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

if [ "${BC_DISABLE_WANDB:-0}" = "1" ]; then
  TRAINER_LOGGER='["console"]'
  probe_msg="wandb disabled by BC_DISABLE_WANDB=1"
elif [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  probe_msg="wandb enabled (key length=${#WANDB_API_KEY})"
else
  TRAINER_LOGGER='["console"]'
  probe_msg="WARNING: no WANDB_API_KEY in env — training will only log to console"
fi

BC_VAL_MAX_SAMPLES=${BC_VAL_MAX_SAMPLES:--1}
BC_PROMPT_LENGTH=${BC_PROMPT_LENGTH:-8192}
BC_CONTEXT_LENGTH=${BC_CONTEXT_LENGTH:-40960}
BC_RESPONSE_LENGTH=${BC_RESPONSE_LENGTH:-$((BC_CONTEXT_LENGTH - BC_PROMPT_LENGTH))}
BC_MAX_TOKEN_LEN_PER_GPU=${BC_MAX_TOKEN_LEN_PER_GPU:-$((BC_PROMPT_LENGTH + BC_RESPONSE_LENGTH))}
BC_MAX_TURN=${BC_MAX_TURN:-100}
BC_SESSION_TIMEOUT=${BC_SESSION_TIMEOUT:-600}
BC_TURN_MAX_NEW_TOKENS=${BC_TURN_MAX_NEW_TOKENS:-768}
BC_FINAL_ANSWER_RESERVE=${BC_FINAL_ANSWER_RESERVE:-1024}
BC_MAX_SESSION=${BC_MAX_SESSION:-10}
BC_SEARCH_TIMEOUT_SECONDS=${BC_SEARCH_TIMEOUT_SECONDS:-600}
BC_METHOD=${BC_METHOD:-baseline}
BC_EXPERIMENT_MODEL_TAG=${BC_EXPERIMENT_MODEL_TAG:-8b}
BC_CTXGRAPH_PROTOCOL=${BC_CTXGRAPH_PROTOCOL:-legacy}
BC_CONTROLLER_ACTION_POLICY=${BC_CONTROLLER_ACTION_POLICY:-structural}

case "$BC_EXPERIMENT_MODEL_TAG" in
  *[!A-Za-z0-9_-]*)
    echo "ERROR: BC_EXPERIMENT_MODEL_TAG may contain only letters, numbers, underscores, and hyphens"
    exit 1
    ;;
esac
case "$BC_CONTROLLER_ACTION_POLICY" in
  balanced|structural) ;;
  *)
    echo "ERROR: BC_CONTROLLER_ACTION_POLICY must be balanced or structural; got $BC_CONTROLLER_ACTION_POLICY"
    exit 1
    ;;
esac
case "$BC_CTXGRAPH_PROTOCOL" in
  legacy)
    BC_STRUCTURED_GRAPH_CONTROLLER=false
    BC_CONTROLLER_OWNED_TOOL_FORMATTING=false
    ;;
  controller)
    if [ "$BC_METHOD" != "contextgraph" ]; then
      echo "ERROR: BC_CTXGRAPH_PROTOCOL=controller requires BC_METHOD=contextgraph"
      exit 1
    fi
    BC_STRUCTURED_GRAPH_CONTROLLER=true
    BC_CONTROLLER_OWNED_TOOL_FORMATTING=true
    ;;
  *)
    echo "ERROR: BC_CTXGRAPH_PROTOCOL must be legacy or controller; got $BC_CTXGRAPH_PROTOCOL"
    exit 1
    ;;
esac

if [ "$BC_RESPONSE_LENGTH" -le 0 ]; then
  echo "ERROR: BC_RESPONSE_LENGTH must be positive (context=$BC_CONTEXT_LENGTH prompt=$BC_PROMPT_LENGTH)"
  exit 1
fi

case "$BC_METHOD" in
  baseline)
    TRAIN_MODULE=scripts.train_baseline
    AGENT_LOOP=react_agent
    WORKFLOW=search_base
    PROCESS_REWARD='[flat]'
    LAMBDA_COST=0.002
    CONSOLIDATION_INTERVAL=0
    ;;
  foldagent)
    TRAIN_MODULE=scripts.train_fold
    AGENT_LOOP=fold_agent
    WORKFLOW=search_branch
    PROCESS_REWARD='[flat,scope]'
    LAMBDA_COST=0.002
    CONSOLIDATION_INTERVAL=0
    ;;
  contextgraph)
    TRAIN_MODULE=scripts.train_graph
    AGENT_LOOP=context_graph_isolated_agent
    WORKFLOW=search_graph
    PROCESS_REWARD='[flat,scope,graph]'
    LAMBDA_COST=0.02
    CONSOLIDATION_INTERVAL=5
    ;;
  *)
    echo "ERROR: BC_METHOD must be one of: baseline, foldagent, contextgraph (got '$BC_METHOD')"
    exit 1
    ;;
esac

# Qwen3-8B's config advertises 40,960 positions.  Longer runs must override
# both the HF actor config and vLLM's independently loaded HF config.
LONG_CONTEXT_ARGS=()
if [ "$BC_MAX_TOKEN_LEN_PER_GPU" -gt 40960 ]; then
  BC_YARN_FACTOR=${BC_YARN_FACTOR:-2.0}
  BC_YARN_ORIGINAL_LENGTH=${BC_YARN_ORIGINAL_LENGTH:-32768}
  LONG_CONTEXT_OVERRIDE="{max_position_embeddings:${BC_MAX_TOKEN_LEN_PER_GPU},rope_scaling:{rope_type:yarn,factor:${BC_YARN_FACTOR},original_max_position_embeddings:${BC_YARN_ORIGINAL_LENGTH}}}"
  LONG_CONTEXT_ARGS+=(
    "+actor_rollout_ref.model.override_config=${LONG_CONTEXT_OVERRIDE}"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides=${LONG_CONTEXT_OVERRIDE}"
  )
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_${BC_METHOD}_${BC_CTXGRAPH_PROTOCOL}_bc_${BC_EXPERIMENT_MODEL_TAG}_4n_zeroshot_${BC_MAX_TOKEN_LEN_PER_GPU}ctx_${TS}}

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  ZERO-SHOT EVAL: $BC_METHOD on BrowseComp-Plus test split (8B, 4 nodes, val_only=True)"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Worker(s): ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  Embedder model: $EMBED_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Logger: ${probe_msg}"
echo "  Caps: val_samples=$BC_VAL_MAX_SAMPLES prompt=$BC_PROMPT_LENGTH response=$BC_RESPONSE_LENGTH total_context=$BC_MAX_TOKEN_LEN_PER_GPU max_turn=$BC_MAX_TURN final_answer_reserve=$BC_FINAL_ANSWER_RESERVE"
echo "  Graph protocol: $BC_CTXGRAPH_PROTOCOL structured_controller=$BC_STRUCTURED_GRAPH_CONTROLLER controller_formatting=$BC_CONTROLLER_OWNED_TOOL_FORMATTING action_policy=$BC_CONTROLLER_ACTION_POLICY"
if [ ${#LONG_CONTEXT_ARGS[@]} -gt 0 ]; then
  echo "  Long context: YaRN factor=$BC_YARN_FACTOR original=$BC_YARN_ORIGINAL_LENGTH (HF actor + vLLM)"
fi
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: BrowseComp data parquets + HF datasets must exist ──
probe "checking BrowseComp artefacts"
TRAIN_PARQUET="$PROJECT_ROOT/data/bc_train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data/bc_test.parquet"
for f in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    echo "       Run: gdown 'https://drive.google.com/uc?id=1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM' -O bc.zip && unzip bc.zip -d data/"
    exit 1
  fi
done
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
if [ "$BC_STRUCTURED_GRAPH_CONTROLLER" = "true" ]; then
  probe "checking vLLM guided-decoding support"
  python -c "from agents.graph_controller import merge_decision_schema; from vllm import SamplingParams; from vllm.sampling_params import GuidedDecodingParams; p=SamplingParams(guided_decoding=GuidedDecodingParams(json=merge_decision_schema([0,1]))); assert p.guided_decoding.json; print('vLLM guided decoding: ok')"
fi

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

# ── Topology: NODE0 = dedicated search server (no Ray), NODE1 = Ray head, NODE2-3 = workers ──
SEARCH_NODE=${NODELIST[0]}
SEARCH_NODE_IP=$(getent hosts "$SEARCH_NODE" | awk '{print $1}')
TRAINER_HEAD_NODE=${NODELIST[1]}
TRAINER_HEAD_IP=$(getent hosts "$TRAINER_HEAD_NODE" | awk '{print $1}')
echo "  Dedicated search node: $SEARCH_NODE ($SEARCH_NODE_IP)"
echo "  Trainer Ray head:      $TRAINER_HEAD_NODE ($TRAINER_HEAD_IP)"
echo "  Trainer workers:       ${NODELIST[@]:2}"

# ── Start envs/search_server.py on dedicated SEARCH_NODE ──
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
  if curl -fsS "http://${SEARCH_NODE_IP}:18999/health" >/dev/null 2>&1; then
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

probe "waiting for search server /search probe (up to ${BC_SEARCH_TIMEOUT_SECONDS}s)"
SEARCH_OK=0
for _ in $(seq 1 "$BC_SEARCH_TIMEOUT_SECONDS"); do
  if curl -fsS -X POST -H 'Content-Type: application/json' \
      -d '{"query":"Eiffel Tower","k":1}' \
      "http://${SEARCH_NODE_IP}:18999/search" >/dev/null 2>&1; then
    SEARCH_OK=1
    break
  fi
  if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
    echo "ERROR: search server exited before /search probe succeeded. Last 80 lines:"
    tail -80 /tmp/hp_server_$$.log || true
    exit 1
  fi
  sleep 1
done
if [ "$SEARCH_OK" != "1" ]; then
  echo "ERROR: search server not reachable at ${SEARCH_NODE_IP}:18999. Last 80 lines:"
  tail -80 /tmp/hp_server_$$.log || true
  kill "$SEARCH_PID" 2>/dev/null || true
  exit 1
fi
export LOCAL_SEARCH_URL="http://${SEARCH_NODE_IP}:18999"
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
  kill "$SEARCH_PID" 2>/dev/null || true
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
echo "  Launching $BC_METHOD ZERO-SHOT eval (4 nodes, total_context=$BC_MAX_TOKEN_LEN_PER_GPU, val_samples=$BC_VAL_MAX_SAMPLES)"
echo "  agent_loop=$AGENT_LOOP workflow=$WORKFLOW process_reward=$PROCESS_REWARD"
echo "  vLLM gpu_memory_utilization=0.6 + FSDP CPU offload"
echo "  val_only=True (one val pass on bc_test.parquet then exit; no training)"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" --chdir="$PROJECT_ROOT" \
  --export=ALL,LOCAL_SEARCH_URL="$LOCAL_SEARCH_URL",OPENAI_API_KEY="$OPENAI_API_KEY",JUDGE_MODEL="$JUDGE_MODEL" \
  python -m "$TRAIN_MODULE" \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop="$AGENT_LOOP" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  "${LONG_CONTEXT_ARGS[@]}" \
  actor_rollout_ref.rollout.prompt_length=${BC_PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${BC_RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${BC_MAX_TOKEN_LEN_PER_GPU} \
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
  data.train_files=data/bc_train.parquet \
  data.val_files=data/bc_test.parquet \
  data.train_batch_size=12 \
  data.max_prompt_length=${BC_PROMPT_LENGTH} \
  data.max_response_length=${BC_RESPONSE_LENGTH} \
  data.val_max_samples=${BC_VAL_MAX_SAMPLES} \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=12 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${BC_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${BC_MAX_TOKEN_LEN_PER_GPU} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow="$WORKFLOW" \
  +actor_rollout_ref.rollout.plugin.max_turn=${BC_MAX_TURN} \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=${BC_TURN_MAX_NEW_TOKENS} \
  +actor_rollout_ref.rollout.plugin.max_session=${BC_MAX_SESSION} \
  +actor_rollout_ref.rollout.plugin.val_max_session=${BC_MAX_SESSION} \
  +actor_rollout_ref.rollout.plugin.session_timeout=${BC_SESSION_TIMEOUT} \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=${BC_RESPONSE_LENGTH} \
  +actor_rollout_ref.rollout.plugin.process_reward="$PROCESS_REWARD" \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.2 \
  +actor_rollout_ref.rollout.plugin.lambda_cost="$LAMBDA_COST" \
  +actor_rollout_ref.rollout.plugin.consolidation_interval="$CONSOLIDATION_INTERVAL" \
  +actor_rollout_ref.rollout.plugin.structured_graph_controller=$BC_STRUCTURED_GRAPH_CONTROLLER \
  +actor_rollout_ref.rollout.plugin.controller_owned_tool_formatting=$BC_CONTROLLER_OWNED_TOOL_FORMATTING \
  +actor_rollout_ref.rollout.plugin.controller_action_policy=$BC_CONTROLLER_ACTION_POLICY \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_max_turn=${BC_MAX_TURN} \
  +actor_rollout_ref.rollout.plugin.val_response_length=${BC_RESPONSE_LENGTH} \
  +actor_rollout_ref.rollout.plugin.final_answer_reserve=${BC_FINAL_ANSWER_RESERVE} \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=$((NUM_NODES - 1)) \
  trainer.total_training_steps=1 \
  trainer.test_freq=999 \
  trainer.save_freq=999 \
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
