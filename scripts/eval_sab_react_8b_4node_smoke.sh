#!/bin/bash
#SBATCH -J eval-sab-react-8b-4n-smoke
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/eval-sab-react-8b-4n-smoke.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/eval-sab-react-8b-4n-smoke.%j.err
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 1:30:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 1.5-hour 4-NODE SMOKE eval for ScienceAgentBench vanilla ReAct
# @ Qwen3-8B (dense). Smoke variant — runs the full 102-task pipeline on
# a smaller topology to validate end-to-end wiring before committing to
# the 5-node x 3-method x 2h paper-grade pass.
#
# Differences vs eval_sab_react_8b_5node_2h.sh:
#   #SBATCH -N            5    -> 4
#   #SBATCH -t            2h   -> 1.5h
#   NUM_NODES check       5    -> 4
#   trainer.nnodes        5    -> 4  (NUM_NODES — automatic)
#   data.train_batch_size 8    -> 4  (4 trainer GPUs, divisibility-clean;
#                                     val_only=True so this only affects
#                                     the unused train loop init path)
#   ppo_mini_batch_size   8    -> 4
#   EXPERIMENT_NAME / job name appended with "_smoke"
#
# Can also be invoked from inside an idev -N 4 session via
#   bash scripts/eval_sab_react_8b_4node_smoke.sh
# because the #SBATCH lines are ignored when not run under sbatch and
# all node info is read from SLURM env vars at runtime.
#
# Topology (5 nodes, no search server):
#   NODELIST[0]   = Ray head + trainer rank 0
#   NODELIST[1-4] = Ray workers (4 FSDP trainer GPUs total = 1 head + 3)
#   No dedicated search node — SAB uses an in-process Python sandbox
#   (envs/scienceagent_sandbox.py), not an external service.
#
# Expected wall clock:
#   model init        ~3-5 min
#   greedy val rollout ~60-90 min (102 tasks * ~30-50s LLM + sandbox exec,
#                                   parallelized across 4 trainer GPUs)
#   no LLM judge (file-existence scorer in env.get_reward; Phase D2)
#   total: ~75-100 min, padded to 2h.
#
# Pre-flight (one-time, login node):
#   # Download HF CSV
#   mkdir -p data
#   wget https://huggingface.co/datasets/osunlp/ScienceAgentBench/resolve/main/ScienceAgentBench.csv \
#        -O data/ScienceAgentBench.csv
#   # Unpack the full benchmark zip (password-protected — request from upstream)
#   #   benchmark zip -> data/sab_benchmark/{datasets,eval_programs,gold_programs}/
#   unzip benchmark.zip -d data/sab_benchmark
#   # Build per-workflow parquets
#   python scripts/make_sab_data.py \
#       --csv data/ScienceAgentBench.csv \
#       --benchmark-dir data/sab_benchmark \
#       --out-dir data
#   # ensure the model is cached:
#   hf download Qwen/Qwen3-8B
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Self-owned log: line-buffered, written directly to Lustre so early output
#    survives even if slurmstepd's stdout buffer is lost on a hard kill / node fail.
#    Also records the submit-dir pwd to catch wrong-WorkDir relative -o failures.
LOG_ROOT=/work/09281/chc_1996/vista/context-graph/logs
SUBMIT_LOG_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}/logs"
EXTRA_LOG_ROOT="${EXTRA_LOG_ROOT:-}"
mkdir -p "$LOG_ROOT" "$SUBMIT_LOG_ROOT"
if [ -n "$EXTRA_LOG_ROOT" ]; then mkdir -p "$EXTRA_LOG_ROOT"; fi
SELF_LOG_NAME="${SLURM_JOB_NAME:-sab}.${SLURM_JOB_ID:-local}.self.log"
TEE_TARGETS=("$LOG_ROOT/$SELF_LOG_NAME")
if [ "$SUBMIT_LOG_ROOT" != "$LOG_ROOT" ]; then TEE_TARGETS+=("$SUBMIT_LOG_ROOT/$SELF_LOG_NAME"); fi
if [ -n "$EXTRA_LOG_ROOT" ] && [ "$EXTRA_LOG_ROOT" != "$LOG_ROOT" ] && [ "$EXTRA_LOG_ROOT" != "$SUBMIT_LOG_ROOT" ]; then TEE_TARGETS+=("$EXTRA_LOG_ROOT/$SELF_LOG_NAME"); fi
SAB_FILTER_TEARDOWN_NOISE=${SAB_FILTER_TEARDOWN_NOISE:-1}
if [ "$SAB_FILTER_TEARDOWN_NOISE" = "1" ]; then
  exec > >(stdbuf -oL awk '
    /validation generation end|EVAL RUN COMPLETED|EVAL RUN FAILED/ { post=1; skip=0 }
    post && /Exception ignored in atexit callback: <function _start_and_connect_service/ { skip=1; next }
    post && /Exception in thread MPClientEngineMonitor:/ { skip=1; next }
    post && /BrokenPipeError: \[Errno 32\] Broken pipe/ { next }
    post && /RuntimeError: There is no current event loop in thread '\''MPClientEngineMonitor'\''/ { next }
    post && /No running event loop\. zmq\.asyncio should be used from within an asyncio loop\./ { next }
    post && /Engine core proc EngineCore_0 died unexpectedly, shutting down client\./ { next }
    skip && /^\([^)]*(TaskRunner|vLLMHttpServer|pid=|WorkerDict)/ {
      if ($0 ~ /(wandb\/sdk\/lib\/service|wandb\/sdk\/lib\/asyncio_manager|asyncio\/streams\.py|concurrent\/futures\/_base\.py|vllm\/v1\/engine\/core_client\.py|weakref\.py|zmq\/_future\.py|zmq\/asyncio\.py|uvloop\/__init__\.py|threading\.py|Traceback \(most recent call last\)|return info\.func|self\._target|self\.run\(\)|_self\.shutdown\(\)|self\._finalizer\(\)|loop = self\.output_socket|current_loop = self\._default_loop\(\)|return asyncio\.get_event_loop\(\)|raise RuntimeError|await self\._writer\.wait_closed\(\)|await self\._client\.close\(\)|raise self\._exception)/) next
      skip=0
    }
    { print; fflush() }
  ' | tee -a "${TEE_TARGETS[@]}") 2>&1
else
  exec > >(stdbuf -oL tee -a "${TEE_TARGETS[@]}") 2>&1
fi
echo "+++ [self-log] host=$(hostname -s) date=$(date) job=${SLURM_JOB_ID:-NA} submit_pwd=$(pwd) log_targets=${TEE_TARGETS[*]}"

# ── Vista cache redirects (avoid NFS flock) ──
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-600}
RAY_HEAD_SETTLE_SECONDS=${RAY_HEAD_SETTLE_SECONDS:-60}
RAY_WORKER_STAGGER_SECONDS=${RAY_WORKER_STAGGER_SECONDS:-8}
RAY_CLUSTER_SETTLE_SECONDS=${RAY_CLUSTER_SETTLE_SECONDS:-60}
RAY_STATUS_TIMEOUT_SECONDS=${RAY_STATUS_TIMEOUT_SECONDS:-900}
RAY_STATUS_POLL_SECONDS=${RAY_STATUS_POLL_SECONDS:-10}
RAY_PORT=${RAY_PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 20000)))}
RAY_TMPDIR_ROOT=${RAY_TMPDIR_ROOT:-/tmp/ray_context_graph_${USER:-unknown}_${SLURM_JOB_ID:-local_$$}}
case "$RAY_TMPDIR_ROOT" in
  /tmp/ray_context_graph_*) ;;
  *)
    echo "ERROR: refusing unsafe RAY_TMPDIR_ROOT=$RAY_TMPDIR_ROOT"
    exit 1
    ;;
esac

# Real per-task eval (eval_programs/<script> -> [0,1] success) vs the
# Phase-D2 file-existence placeholder. Formal runs default to the real
# evaluator and reject an explicit attempt to disable it.
SAB_RUN_TAG=${SAB_RUN_TAG:-smoke}
if [ "$SAB_RUN_TAG" = "formal" ]; then
  export SAB_REAL_EVAL=${SAB_REAL_EVAL:-1}
  SAB_DUMP_VALIDATION=${SAB_DUMP_VALIDATION:-1}
else
  export SAB_REAL_EVAL=${SAB_REAL_EVAL:-0}
  SAB_DUMP_VALIDATION=${SAB_DUMP_VALIDATION:-0}
fi
export SAB_EXPOSE_EVAL_CONTRACT=${SAB_EXPOSE_EVAL_CONTRACT:-0}
export SAB_INTERACTIVE_EVAL_FEEDBACK=${SAB_INTERACTIVE_EVAL_FEEDBACK:-0}
export SAB_DEBUG_IO=${SAB_DEBUG_IO:-0}
export SAB_NO_OUTPUT_HINT_AFTER=${SAB_NO_OUTPUT_HINT_AFTER:-2}
SAB_LOG_VAL_GENERATIONS=${SAB_LOG_VAL_GENERATIONS:-0}
case "$SAB_RUN_TAG" in
  *[!A-Za-z0-9_-]*)
    echo "ERROR: SAB_RUN_TAG may contain only letters, numbers, underscores, and hyphens"
    exit 1
    ;;
esac
if [ "$SAB_RUN_TAG" = "formal" ] && [ "$SAB_REAL_EVAL" != "1" ]; then
  echo "ERROR: SAB_RUN_TAG=formal requires SAB_REAL_EVAL=1"
  exit 1
fi

# ── WANDB ──
if [ -n "${WORK:-}" ] && [ -f "$WORK/.openai_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.openai_env"
fi
if [ -n "${WORK:-}" ] && [ -f "$WORK/.wandb_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.wandb_env"
fi
export WANDB_API_KEY=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

# ── Conda + CUDA ──
set +u  # conda activation scripts reference unbound vars (PS1, _CE_CONDA) -> set -u would kill us silently
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
set -u
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
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Per-trajectory sandbox workdir root (scratch is fastest on Vista)
export SAB_WORKDIR_ROOT=${SAB_WORKDIR_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/sab_workdirs}
mkdir -p "$SAB_WORKDIR_ROOT"

# ── Node info ──
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}

if [ "$NUM_NODES" -ne 4 ]; then
  echo "Expected 4 nodes (set #SBATCH -N 4 or use idev -N 4), got $NUM_NODES"
  exit 1
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$NUM_NODES}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:--1}
SAB_TRAIN_MAX_SAMPLES=${SAB_TRAIN_MAX_SAMPLES:--1}
SAB_PROMPT_LENGTH=${SAB_PROMPT_LENGTH:-16384}
if [ "$SAB_RUN_TAG" = "formal" ]; then
  SAB_RESPONSE_LENGTH=${SAB_RESPONSE_LENGTH:-24576}
  SAB_MAX_TOKEN_LEN_PER_GPU=${SAB_MAX_TOKEN_LEN_PER_GPU:-40960}
  SAB_VAL_MAX_TURN=${SAB_VAL_MAX_TURN:-32}
  SAB_TURN_MAX_NEW_TOKENS=${SAB_TURN_MAX_NEW_TOKENS:-2048}
else
  SAB_RESPONSE_LENGTH=${SAB_RESPONSE_LENGTH:-12288}
  SAB_MAX_TOKEN_LEN_PER_GPU=${SAB_MAX_TOKEN_LEN_PER_GPU:-$((SAB_PROMPT_LENGTH + SAB_RESPONSE_LENGTH))}
  SAB_VAL_MAX_TURN=${SAB_VAL_MAX_TURN:-24}
  SAB_TURN_MAX_NEW_TOKENS=${SAB_TURN_MAX_NEW_TOKENS:-512}
fi
SAB_METHOD=${SAB_METHOD:-react}
case "$SAB_METHOD" in
  react)
    SAB_METHOD_LABEL=ReAct
    SAB_AGENT_LOOP=react_agent_code
    SAB_WORKFLOW=code
    SAB_PROCESS_REWARD='[flat]'
    SAB_DATA_FILE=data/sab_test_code.parquet
    SAB_LAMBDA_COST=0.002
    ;;
  fold)
    SAB_METHOD_LABEL=FoldAgent
    SAB_AGENT_LOOP=fold_agent_code
    SAB_WORKFLOW=code_branch
    SAB_PROCESS_REWARD='[flat,scope]'
    SAB_DATA_FILE=data/sab_test_code_branch.parquet
    SAB_LAMBDA_COST=0.002
    ;;
  ctxgraph)
    SAB_METHOD_LABEL=ContextGraph
    SAB_AGENT_LOOP=context_graph_code_isolated_agent
    SAB_WORKFLOW=code_graph
    SAB_PROCESS_REWARD='[flat,scope,graph]'
    SAB_DATA_FILE=data/sab_test_code_graph.parquet
    SAB_LAMBDA_COST=0.02
    ;;
  *)
    echo "ERROR: SAB_METHOD must be react, fold, or ctxgraph; got $SAB_METHOD"
    exit 1
    ;;
esac
SAB_MAX_SESSION=${SAB_MAX_SESSION:-4}
if [ "$SAB_RUN_TAG" = "formal" ]; then
  SAB_BRANCH_LEN=${SAB_BRANCH_LEN:-32768}
else
  SAB_BRANCH_LEN=${SAB_BRANCH_LEN:-$SAB_RESPONSE_LENGTH}
fi
if (( TRAIN_BATCH_SIZE % NUM_NODES != 0 )); then
  echo "ERROR: TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE must be divisible by NUM_NODES=$NUM_NODES"
  exit 1
fi
if (( PPO_MINI_BATCH_SIZE % NUM_NODES != 0 )); then
  echo "ERROR: PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE must be divisible by NUM_NODES=$NUM_NODES"
  exit 1
fi

if [ "${SAB_DISABLE_WANDB:-0}" = "1" ]; then
  TRAINER_LOGGER='["console"]'
  probe_msg="wandb disabled by SAB_DISABLE_WANDB=1"
elif [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  probe_msg="wandb enabled (key length=${#WANDB_API_KEY})"
else
  TRAINER_LOGGER='["console"]'
  probe_msg="WARNING: no WANDB_API_KEY in env — eval will only log to console"
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="eval_${SAB_METHOD}_sab_8b_4n_${SAB_RUN_TAG}_${TS}"
TRAINER_DEBUG_OVERRIDES=()
if [ "$SAB_DUMP_VALIDATION" = "1" ]; then
  SAB_VALIDATION_DATA_DIR=${SAB_VALIDATION_DATA_DIR:-${SCRATCH:-/scratch/09281/chc_1996}/sab_validation_generations/$EXPERIMENT_NAME}
  mkdir -p "$SAB_VALIDATION_DATA_DIR"
  TRAINER_DEBUG_OVERRIDES+=("trainer.validation_data_dir=$SAB_VALIDATION_DATA_DIR")
fi
if [ "$SAB_LOG_VAL_GENERATIONS" != "0" ]; then
  TRAINER_DEBUG_OVERRIDES+=("trainer.log_val_generations=$SAB_LOG_VAL_GENERATIONS")
fi

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

wait_for_ray_cluster() {
  local expected_nodes=$1
  local deadline=$((SECONDS + RAY_STATUS_TIMEOUT_SECONDS))
  local status_out active_nodes

  while [ "$SECONDS" -lt "$deadline" ]; do
    status_out=$(ray status 2>&1 || true)
    active_nodes=$(printf '%s\n' "$status_out" | awk '
      /^Active:/ {active=1; next}
      /^Pending:/ {active=0}
      active && /node_/ {count++}
      END {print count + 0}
    ')
    if [ "$active_nodes" -ge "$expected_nodes" ]; then
      printf '%s\n' "$status_out"
      return 0
    fi
    printf '+++ [%s] waiting for Ray nodes: active=%s expected=%s\n' \
      "$(date +%H:%M:%S)" "$active_nodes" "$expected_nodes"
    sleep "$RAY_STATUS_POLL_SECONDS"
  done

  echo "ERROR: Ray cluster did not reach $expected_nodes active nodes within ${RAY_STATUS_TIMEOUT_SECONDS}s."
  echo "Last ray status output:"
  printf '%s\n' "$status_out"
  return 1
}

echo "=============================================================="
echo "  ZERO-SHOT EVAL: $SAB_METHOD_LABEL ($SAB_WORKFLOW) on ScienceAgentBench (Qwen3-8B dense, 4 nodes ${SAB_RUN_TAG^^}, val_only=True)"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Workers: ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  Experiment:     $EXPERIMENT_NAME"
echo "  Sandbox workdir root: $SAB_WORKDIR_ROOT"
echo "  Logger: ${probe_msg}"
echo "  Batch sizes:    train=$TRAIN_BATCH_SIZE ppo_mini=$PPO_MINI_BATCH_SIZE"
echo "  Sample caps:    val=$SAB_VAL_MAX_SAMPLES train=$SAB_TRAIN_MAX_SAMPLES"
echo "  Token caps:     prompt=$SAB_PROMPT_LENGTH response=$SAB_RESPONSE_LENGTH max_token_gpu=$SAB_MAX_TOKEN_LEN_PER_GPU"
echo "  Turn caps:      val_max_turn=$SAB_VAL_MAX_TURN turn_max_new_tokens=$SAB_TURN_MAX_NEW_TOKENS"
echo "  Method config:  agent_loop=$SAB_AGENT_LOOP workflow=$SAB_WORKFLOW process_reward=$SAB_PROCESS_REWARD"
echo "  SAB_REAL_EVAL:  $SAB_REAL_EVAL"
echo "  Ray bootstrap:  port=$RAY_PORT raylet_wait=${RAY_raylet_start_wait_time_s}s status_timeout=${RAY_STATUS_TIMEOUT_SECONDS}s tmp=$RAY_TMPDIR_ROOT"
echo "  Debug:          SAB_DEBUG_IO=$SAB_DEBUG_IO SAB_DUMP_VALIDATION=$SAB_DUMP_VALIDATION no_output_hint_after=$SAB_NO_OUTPUT_HINT_AFTER ${SAB_VALIDATION_DATA_DIR:+validation_dir=$SAB_VALIDATION_DATA_DIR}"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: data parquets must exist ──
probe "checking ScienceAgentBench artefacts"
VAL_PARQUET="$PROJECT_ROOT/$SAB_DATA_FILE"
if [ ! -f "$VAL_PARQUET" ]; then
  echo "ERROR: missing $VAL_PARQUET"
  echo "       Run: python scripts/make_sab_data.py --csv data/ScienceAgentBench.csv \\"
  echo "                 --benchmark-dir data/sab_benchmark --out-dir data"
  exit 1
fi
# verl wants a train_files path too even with val_only=True; reuse the same file.
TRAIN_PARQUET="$VAL_PARQUET"
probe "SAB parquet: $VAL_PARQUET"

if [ "$SAB_REAL_EVAL" = "1" ]; then
  if [ ! -f "$PROJECT_ROOT/gpt4_visual_judge.py" ]; then
    echo "ERROR: missing $PROJECT_ROOT/gpt4_visual_judge.py"
    echo "       Visual SAB evaluators cannot run without this helper."
    exit 1
  fi
  if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${AZURE_OPENAI_KEY:-}" ]; then
    echo "ERROR: SAB_REAL_EVAL=1 requires OPENAI_API_KEY or Azure OpenAI credentials"
    echo "       because SAB contains GPT-judged visualization tasks."
    exit 1
  fi
  probe "real evaluator helpers + visual judge credentials ok"
fi

# ── Pre-flight: model weights must be cached (offline) ──
probe "checking model cache"
TRAINER_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
if [ ! -d "$TRAINER_CACHE_DIR" ]; then
  echo "ERROR: $MODEL_PATH not cached at $TRAINER_CACHE_DIR"
  echo "       Login node: hf download $MODEL_PATH"
  exit 1
fi
probe "trainer cache: $TRAINER_CACHE_DIR"

# ── Topology: NODE0 = Ray head + trainer rank 0; NODE1-4 = Ray workers ──
TRAINER_HEAD_NODE=${NODELIST[0]}
TRAINER_HEAD_IP=$(getent hosts "$TRAINER_HEAD_NODE" | awk '{print $1}')
echo "  Trainer Ray head:      $TRAINER_HEAD_NODE ($TRAINER_HEAD_IP)"
echo "  Trainer workers:       ${NODELIST[@]:1}"

# ── Stale Ray cleanup on all nodes ──
probe "ray stop sweep across $NUM_NODES nodes"
for node in "${NODELIST[@]}"; do
  srun --overlap --nodes=1 --ntasks=1 -w "$node" bash -c '
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    ray stop -f >/dev/null 2>&1 || true
    case "'"$RAY_TMPDIR_ROOT"'" in /tmp/ray_context_graph_*) rm -rf "'"$RAY_TMPDIR_ROOT"'" || true ;; esac
    mkdir -p "'"$RAY_TMPDIR_ROOT"'"
  ' || true
done
sleep 5
probe "ray stop sweep done"

# ── Sanity imports ──
probe "python sanity imports"
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
python -c "from envs.scienceagent_sandbox import CodeSandbox; from envs.scienceagent_env import ScienceAgentEnv; print('SAB env OK')"
probe "sanity imports done"

# ── Ray head on NODELIST[0] ──
probe "starting Ray head on $TRAINER_HEAD_NODE"
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
  export SAB_WORKDIR_ROOT='"$SAB_WORKDIR_ROOT"'
  export SAB_REAL_EVAL='"$SAB_REAL_EVAL"'
  export SAB_EXPOSE_EVAL_CONTRACT='"$SAB_EXPOSE_EVAL_CONTRACT"'
  export SAB_INTERACTIVE_EVAL_FEEDBACK='"$SAB_INTERACTIVE_EVAL_FEEDBACK"'
  export SAB_DEBUG_IO='"$SAB_DEBUG_IO"'
  export RAY_raylet_start_wait_time_s='"$RAY_raylet_start_wait_time_s"'
  ray start --head --node-ip-address='"$TRAINER_HEAD_IP"' --port='"$RAY_PORT"' --temp-dir='"$RAY_TMPDIR_ROOT"' \
    --num-cpus=70 --num-gpus=1 --include-dashboard=false --disable-usage-stats --block
' &
RAY_HEAD_PID=$!
sleep "$RAY_HEAD_SETTLE_SECONDS"
if ! kill -0 "$RAY_HEAD_PID" 2>/dev/null; then
  echo "ERROR: Ray head process exited before workers could join."
  wait "$RAY_HEAD_PID" || true
  exit 1
fi
probe "Ray head sleep done; launching $((NUM_NODES - 1)) trainer workers"

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
    export SAB_WORKDIR_ROOT='"$SAB_WORKDIR_ROOT"'
    export SAB_REAL_EVAL='"$SAB_REAL_EVAL"'
    export SAB_EXPOSE_EVAL_CONTRACT='"$SAB_EXPOSE_EVAL_CONTRACT"'
    export SAB_INTERACTIVE_EVAL_FEEDBACK='"$SAB_INTERACTIVE_EVAL_FEEDBACK"'
    export SAB_DEBUG_IO='"$SAB_DEBUG_IO"'
    export RAY_raylet_start_wait_time_s='"$RAY_raylet_start_wait_time_s"'
    ray start --address='"${TRAINER_HEAD_IP}:${RAY_PORT}"' --temp-dir='"$RAY_TMPDIR_ROOT"' --num-cpus=70 --num-gpus=1 --block
  ' &
  WORKER_PIDS+=("$!")
  sleep "$RAY_WORKER_STAGGER_SECONDS"
done
sleep "$RAY_CLUSTER_SETTLE_SECONDS"
probe "all $((NUM_NODES - 1)) trainer workers launched, cluster settling"

cleanup() {
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  for pid in "${WORKER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

export RAY_ADDRESS=${TRAINER_HEAD_IP}:${RAY_PORT}
probe "waiting for Ray cluster readiness"
if ! wait_for_ray_cluster "$NUM_NODES"; then
  echo "Ray temp/session dir: $RAY_TMPDIR_ROOT"
  find "$RAY_TMPDIR_ROOT" -maxdepth 3 -type f \( -name 'gcs_server*.out' -o -name 'gcs_server*.err' -o -name 'raylet*.out' -o -name 'raylet*.err' -o -name 'dashboard*.log' -o -name 'dashboard*.err' \) -print 2>/dev/null | tail -40 || true
  echo "ERROR: Ray cluster did not start cleanly; aborting before trainer launch."
  exit 1
fi

echo "=============================================================="
echo "  Launching $SAB_METHOD_LABEL ($SAB_WORKFLOW) ZERO-SHOT eval (4 nodes ${SAB_RUN_TAG^^}, ScienceAgentBench test cap=$SAB_VAL_MAX_SAMPLES)"
echo "  default_agent_loop=$SAB_AGENT_LOOP  workflow=$SAB_WORKFLOW  process_reward=$SAB_PROCESS_REWARD"
echo "  vLLM gpu_memory_utilization=0.6 + FSDP CPU offload (8B fits)"
echo "  val_only=True (one val pass on $SAB_DATA_FILE then exit; no training)"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" --chdir="$PROJECT_ROOT" \
  --export=ALL,SAB_WORKDIR_ROOT="$SAB_WORKDIR_ROOT",SAB_REAL_EVAL="$SAB_REAL_EVAL",SAB_EXPOSE_EVAL_CONTRACT="$SAB_EXPOSE_EVAL_CONTRACT",SAB_INTERACTIVE_EVAL_FEEDBACK="$SAB_INTERACTIVE_EVAL_FEEDBACK",SAB_DEBUG_IO="$SAB_DEBUG_IO",SAB_NO_OUTPUT_HINT_AFTER="$SAB_NO_OUTPUT_HINT_AFTER" \
  python -m scripts.train_sab \
  "${TRAINER_DEBUG_OVERRIDES[@]}" \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=$SAB_AGENT_LOOP \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.prompt_length=$SAB_PROMPT_LENGTH \
  actor_rollout_ref.rollout.response_length=$SAB_RESPONSE_LENGTH \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$SAB_MAX_TOKEN_LEN_PER_GPU \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=1 \
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
  data.train_files=$SAB_DATA_FILE \
  data.val_files=$SAB_DATA_FILE \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.train_max_samples=$SAB_TRAIN_MAX_SAMPLES \
  data.val_max_samples=$SAB_VAL_MAX_SAMPLES \
  data.max_prompt_length=$SAB_PROMPT_LENGTH \
  data.max_response_length=$SAB_RESPONSE_LENGTH \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$SAB_MAX_TOKEN_LEN_PER_GPU \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=$SAB_MAX_TOKEN_LEN_PER_GPU \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=$SAB_WORKFLOW \
  +actor_rollout_ref.rollout.plugin.max_turn=$SAB_VAL_MAX_TURN \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=$SAB_TURN_MAX_NEW_TOKENS \
  +actor_rollout_ref.rollout.plugin.sandbox_timeout=60 \
  +actor_rollout_ref.rollout.plugin.process_reward="$SAB_PROCESS_REWARD" \
  +actor_rollout_ref.rollout.plugin.max_session=$SAB_MAX_SESSION \
  +actor_rollout_ref.rollout.plugin.val_max_session=$SAB_MAX_SESSION \
  +actor_rollout_ref.rollout.plugin.session_timeout=3600 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=$SAB_BRANCH_LEN \
  +actor_rollout_ref.rollout.plugin.max_traj=4 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.2 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=$SAB_LAMBDA_COST \
  +actor_rollout_ref.rollout.plugin.consolidation_interval=5 \
  +actor_rollout_ref.rollout.plugin.uniqueness_weight=0.10 \
  +actor_rollout_ref.rollout.plugin.auto_bind_branch_edges=True \
  +actor_rollout_ref.rollout.plugin.auto_bind_min_overlap=0.05 \
  +actor_rollout_ref.rollout.plugin.val_max_turn=$SAB_VAL_MAX_TURN \
  +actor_rollout_ref.rollout.plugin.val_response_length=$SAB_RESPONSE_LENGTH \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=$NUM_NODES \
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
  echo "  EVAL RUN COMPLETED (exit 0)"
else
  echo "  EVAL RUN FAILED (exit $RC) — check above for first error"
fi
echo "  Finished: $(date)"
echo "=============================================================="

exit $RC
