#!/bin/bash
#SBATCH -J eval-bc-30b-cg-zeroshot
#SBATCH -o logs/eval-bc-30b-cg-zeroshot.%j.out
#SBATCH -e logs/eval-bc-30b-cg-zeroshot.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 2:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 48-hour 8-NODE training for BrowseComp-Plus ContextGraph @ Qwen3-30B-A3B-Thinking-2507
#   8 nodes × 1 GH200 = 8 GPUs (1 dedicated search + 7 trainer, FSDP)
#   FP8 vLLM rollout + BF16 FSDP actor/ref + CPU param/optim offload
#   total_training_steps=30 (matches the 8B 32K paper-match epoch budget;
#   30B step time ~3-4x slower → ~80-100 min/step, 30 step ≈ 40-50h)
#
# Pairs with train_bc_foldagent_30b_8node_48h.sh and
#                train_bc_baseline_30b_8node_48h.sh — same backbone, same
# retrieval substrate, same context budget, only the agent loop differs.
#
# 30B-specific knobs vs the 8B 4-node 32K v3 template:
#   * #SBATCH -N 8 / -t 48:00:00
#   * MODEL_PATH = Qwen/Qwen3-30B-A3B-Thinking-2507 (MoE A3B variant; only
#       3B params activate per token but all 30B must shard across 7 trainer GPUs)
#   * +rollout.quantization=fp8 (vLLM rollout halves weight footprint;
#       FSDP actor stays BF16 for gradient correctness)
#   * gpu_memory_utilization=0.55 (was 0.6 at 8B; give actor more margin
#       since 30B FSDP shard is 60GB/7≈8.6GB params + bigger activation footprint)
#   * train_batch_size = ppo_mini_batch_size = 14 (so 14 × n=8 = 112 ÷ 7
#       trainer = 16/GPU; conservative vs 8B's 32/GPU)
#       ALFWorld 30B template; cuts fragmentation across 32K rollouts)
#
# Topology: NODELIST[0] = dedicated search server (no Ray), NODELIST[1]
# = Ray head + trainer rank 0, NODELIST[2..7] = Ray workers. FSDP across 7 GPUs.
# Production training (30 steps, val + ckpt every 10 steps):
#   - search_server.py loads HF datasets (Tevatron/browsecomp-plus-corpus +
#     precomputed embeddings from miaolu3/browsecomp-plus)
#   - trainer hits OpenAI judge per rollout completion (gpt-5-nano default,
#     override via JUDGE_MODEL env var; cheaper smoke: JUDGE_MODEL=gpt-4o-mini)
#   - ctxgraph agent loop emits graph ops over BC's multi-page browsing tasks
#
# Known 30B risk: Qwen3-30B tokenizer_config.json JSONDecodeError observed on
# job 704096 (ALFWorld). Hypothesised cause is concurrent rank refetches racing
# on the HF cache; HF_HUB_OFFLINE + HF_HUB_DISABLE_FILE_LOCKING are set below
# as the standard mitigation, but the bug is officially undiagnosed.
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
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Thinking-2507}
EMBED_MODEL=${EMBED_MODEL:-Qwen/Qwen3-Embedding-8B}
CTXGRAPH_PROTOCOL=${CTXGRAPH_PROTOCOL:-legacy}
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
case "$CTXGRAPH_PROTOCOL" in
  legacy)
    STRUCTURED_GRAPH_CONTROLLER=false
    CONTROLLER_OWNED_TOOL_FORMATTING=false
    ;;
  controller)
    STRUCTURED_GRAPH_CONTROLLER=true
    CONTROLLER_OWNED_TOOL_FORMATTING=true
    ;;
  *)
    echo "ERROR: CTXGRAPH_PROTOCOL must be legacy or controller; got $CTXGRAPH_PROTOCOL"
    exit 1
    ;;
esac

if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  probe_msg="wandb enabled (key length=${#WANDB_API_KEY})"
else
  TRAINER_LOGGER='["console"]'
  probe_msg="WARNING: no WANDB_API_KEY in env — training will only log to console"
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="eval_ctxgraph_${CTXGRAPH_PROTOCOL}_bc_30b_8n_zeroshot_${TS}"

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  ZERO-SHOT EVAL: ContextGraph on BrowseComp-Plus test split (30B-A3B Thinking, 8 nodes, 32K-resp, FP8 rollout, val_only=True)"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Worker(s): ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  Embedder model: $EMBED_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Logger: ${probe_msg}"
echo "  Graph protocol: $CTXGRAPH_PROTOCOL structured_controller=$STRUCTURED_GRAPH_CONTROLLER controller_formatting=$CONTROLLER_OWNED_TOOL_FORMATTING"
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
if [ "$STRUCTURED_GRAPH_CONTROLLER" = "true" ]; then
  probe "checking vLLM guided-decoding support"
  python -c "from agents.graph_controller import merge_decision_schema; from vllm import SamplingParams; from vllm.sampling_params import GuidedDecodingParams; p=SamplingParams(guided_decoding=GuidedDecodingParams(json=merge_decision_schema([0,1]))); assert p.guided_decoding.json; print('vLLM guided decoding: ok')"
fi
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

probe "waiting for search server /health (up to 240s)"
for i in $(seq 1 120); do
  if curl -fsS "http://${SEARCH_NODE_IP}:18999/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
if ! curl -fsS -X POST -H 'Content-Type: application/json' \
        -d '{"query":"Eiffel Tower","k":1}' \
        "http://${SEARCH_NODE_IP}:18999/search" >/dev/null; then
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
echo "  Launching ContextGraph ZERO-SHOT eval (30B-A3B Thinking, 8 nodes, 32K resp, FP8 rollout, BrowseComp-Plus test=150)"
echo "  v3 reward: continuous concise_main, semi-de-gated graph_shaping (alpha=0.3), lambda_cost=0.02, forced consolidation K=5"
echo "  vLLM gpu_memory_utilization=0.55 + FP8 rollout + FSDP CPU offload"
echo "  val_before_train=True, save_freq=10 (6 ckpts: steps 10/20/.../60), val every 10 steps"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" --chdir="$PROJECT_ROOT" \
  --export=ALL,LOCAL_SEARCH_URL="$LOCAL_SEARCH_URL",OPENAI_API_KEY="$OPENAI_API_KEY",JUDGE_MODEL="$JUDGE_MODEL" \
  python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  +actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=8192 \
  actor_rollout_ref.rollout.response_length=32768 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40960 \
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
  data.train_batch_size=14 \
  data.max_prompt_length=8192 \
  data.max_response_length=32768 \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=14 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40960 \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=40960 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=search_graph \
  +actor_rollout_ref.rollout.plugin.max_turn=100 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=768 \
  +actor_rollout_ref.rollout.plugin.max_session=10 \
  +actor_rollout_ref.rollout.plugin.val_max_session=10 \
  +actor_rollout_ref.rollout.plugin.session_timeout=600 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=32768 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope,graph]' \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.2 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.02 \
  +actor_rollout_ref.rollout.plugin.consolidation_interval=5 \
  +actor_rollout_ref.rollout.plugin.structured_graph_controller=$STRUCTURED_GRAPH_CONTROLLER \
  +actor_rollout_ref.rollout.plugin.controller_owned_tool_formatting=$CONTROLLER_OWNED_TOOL_FORMATTING \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_max_turn=100 \
  +actor_rollout_ref.rollout.plugin.val_response_length=32768 \
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
