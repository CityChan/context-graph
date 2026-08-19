#!/bin/bash
#SBATCH -J eval-sab-react-30b-inst-smoke
#SBATCH -o /work/09281/chc_1996/vista/context-graph/logs/eval-sab-react-30b-inst-smoke.%j.out
#SBATCH -e /work/09281/chc_1996/vista/context-graph/logs/eval-sab-react-30b-inst-smoke.%j.err
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 4:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 4-hour 8-NODE SMOKE eval for ScienceAgentBench vanilla ReAct
# @ Qwen3-30B-A3B-Instruct-2507. Paired with eval_sab_{fold,ctxgraph}_8b_8node_2h.sh
# for a quick 30B Instruct sanity check before attempting a full 102-task run.
#
# Topology (8 nodes, no search server):
#   NODELIST[0]   = Ray head + trainer rank 0
#   NODELIST[1-7] = Ray workers (8 FSDP/vLLM GPUs total = 1 per node)
#   No dedicated search node; SAB uses an in-process Python sandbox
#   (envs/scienceagent_sandbox.py), not an external service.
#
# Expected wall clock:
#   model init        ~3-5 min
#   greedy val rollout defaults to a short diagnostic subset for smoke testing
#   override SAB_VAL_MAX_SAMPLES / SAB_RESPONSE_LENGTH / SAB_VAL_MAX_TURN for larger runs
#   no LLM judge (file-existence scorer in env.get_reward; Phase D2)
#   total: padded to 4h to leave room for 30B model/vLLM initialization.
#
# Pre-flight (one-time, login node):
#   # Download HF CSV
#   mkdir -p data
#   wget https://huggingface.co/datasets/osunlp/ScienceAgentBench/resolve/main/ScienceAgentBench.csv \
#        -O data/ScienceAgentBench.csv
#   # Unpack the full benchmark zip (password-protected; request from upstream)
#   #   benchmark zip -> data/sab_benchmark/{datasets,eval_programs,gold_programs}/
#   unzip benchmark.zip -d data/sab_benchmark
#   # Build per-workflow parquets
#   python scripts/make_sab_data.py \
#       --csv data/ScienceAgentBench.csv \
#       --benchmark-dir data/sab_benchmark \
#       --out-dir data
#   # ensure the model is cached:
#   hf download Qwen/Qwen3-30B-A3B-Instruct-2507
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# The launch plumbing is shared by code-execution benchmarks. Wrappers may
# override these values while the default remains the original SAB behavior.
CODE_BENCHMARK_LABEL=${CODE_BENCHMARK_LABEL:-ScienceAgentBench}
CODE_BENCHMARK_PROFILE=${CODE_BENCHMARK_PROFILE:-sab}
CODE_BENCHMARK_DATA_FILE=${CODE_BENCHMARK_DATA_FILE:-}
CODE_BENCHMARK_TRAIN_MODULE=${CODE_BENCHMARK_TRAIN_MODULE:-scripts.train_sab}
CODE_BENCHMARK_PREPARE_HINT=${CODE_BENCHMARK_PREPARE_HINT:-}
export CODE_BENCHMARK_LABEL CODE_BENCHMARK_PROFILE

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
EARLY_LOG_PATH="$LOG_ROOT/${SELF_LOG_NAME}.early"
printf '+++ [EARLY START] host=%s date=%s job=%s script=%s pwd=%s\n' "$(hostname -s)" "$(date)" "${SLURM_JOB_ID:-NA}" "$0" "$(pwd)" | tee -a "$EARLY_LOG_PATH"
printf '+++ [EARLY START] early_log=%s stdout_target=%s stderr_target=%s\n' "$EARLY_LOG_PATH" "/work/09281/chc_1996/vista/context-graph/logs/eval-sab-react-30b-inst-smoke.${SLURM_JOB_ID:-local}.out" "/work/09281/chc_1996/vista/context-graph/logs/eval-sab-react-30b-inst-smoke.${SLURM_JOB_ID:-local}.err" | tee -a "$EARLY_LOG_PATH"
exec > >(stdbuf -oL tee -a "${TEE_TARGETS[@]}") 2>&1
echo "+++ [self-log] host=$(hostname -s) date=$(date) job=${SLURM_JOB_ID:-NA} submit_pwd=$(pwd) log_targets=${TEE_TARGETS[*]}"
export SELF_LOG_PATH="${TEE_TARGETS[0]}"
echo "+++ [PROGRAM START] eval_sab_react_30b_instruct_8node_smoke.sh"
echo "+++ [PROGRAM START] script=$0"
echo "+++ [PROGRAM START] self_log=$SELF_LOG_PATH"
echo "+++ [PROGRAM START] slurm_job=${SLURM_JOB_ID:-NA} name=${SLURM_JOB_NAME:-NA} nodelist=${SLURM_JOB_NODELIST:-NA}"
echo "+++ [PROGRAM START] git_head=$(git -C /work/09281/chc_1996/vista/context-graph rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "+++ [PROGRAM START] submit_dir=${SLURM_SUBMIT_DIR:-NA} pwd=$(pwd)"

# ── Vista cache redirects (avoid NFS flock) ──
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0

# Real per-task eval (eval_programs/<script> -> [0,1] success) vs the
# Phase-D2 file-existence placeholder. Formal runs require paper-grade scoring
# and preserve per-sample validation generations for audit/re-scoring.
SAB_RUN_TAG=${SAB_RUN_TAG:-smoke}
case "$SAB_RUN_TAG" in
  *[!A-Za-z0-9_-]*)
    echo "ERROR: SAB_RUN_TAG may contain only letters, numbers, underscores, and hyphens"
    exit 1
    ;;
esac
if [ "$SAB_RUN_TAG" = "formal" ]; then
  if [ "$CODE_BENCHMARK_PROFILE" = "sab" ]; then
    export SAB_REAL_EVAL=${SAB_REAL_EVAL:-1}
  else
    export SAB_REAL_EVAL=${SAB_REAL_EVAL:-0}
  fi
  SAB_DUMP_VALIDATION=${SAB_DUMP_VALIDATION:-1}
else
  export SAB_REAL_EVAL=${SAB_REAL_EVAL:-0}
  SAB_DUMP_VALIDATION=${SAB_DUMP_VALIDATION:-0}
fi
if [ "$SAB_RUN_TAG" = "formal" ] && [ "$CODE_BENCHMARK_PROFILE" = "sab" ] && [ "$SAB_REAL_EVAL" != "1" ]; then
  echo "ERROR: SAB_RUN_TAG=formal requires SAB_REAL_EVAL=1"
  exit 1
fi
if [ "$SAB_RUN_TAG" = "formal" ] && [ "$CODE_BENCHMARK_PROFILE" = "discoverybench" ] && [ "${DISCOVERYBENCH_REAL_EVAL:-0}" != "1" ]; then
  echo "ERROR: DiscoveryBench formal runs require DISCOVERYBENCH_REAL_EVAL=1"
  exit 1
fi
export SAB_EXPOSE_EVAL_CONTRACT=${SAB_EXPOSE_EVAL_CONTRACT:-0}
export SAB_INTERACTIVE_EVAL_FEEDBACK=${SAB_INTERACTIVE_EVAL_FEEDBACK:-0}
# Qwen3 Thinking models default to long <think> traces. For tool-use eval,
# disable thinking in the chat template unless explicitly overridden.
export QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-False}

# Load visual-judge credentials before Ray starts. The official SAB evaluator
# uses GPT-4o for visualization tasks, so a real-eval run without credentials
# would otherwise turn an evaluator crash into a misleading task score of zero.
OPENAI_ENV_SOURCE="env"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${AZURE_OPENAI_KEY:-}" ] && [ -z "${AZURE_OPENAI_API_KEY:-}" ]; then
  for openai_env in "${WORK:-}/.openai_env" /work/09281/chc_1996/vista/.openai_env "$HOME/.openai_env"; do
    if [ -n "$openai_env" ] && [ -f "$openai_env" ]; then
      # shellcheck disable=SC1090
      source "$openai_env"
      OPENAI_ENV_SOURCE="$openai_env"
      break
    fi
  done
fi

# WANDB: source credentials from explicit env first, then common Vista/project locations.
WANDB_ENV_SOURCE="env"
if [ -z "${WANDB_API_KEY:-}" ]; then
  for wandb_env in "${WORK:-}/.wandb_env" /work/09281/chc_1996/vista/.wandb_env /work/09281/chc_1996/vista/context-graph/.wandb_env "$HOME/.wandb_env"; do
    if [ -n "$wandb_env" ] && [ -f "$wandb_env" ]; then
      # shellcheck disable=SC1090
      source "$wandb_env"
      WANDB_ENV_SOURCE="$wandb_env"
      break
    fi
  done
fi
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_DIR=${WANDB_DIR:-/work/09281/chc_1996/vista/context-graph/wandb}
mkdir -p "$WANDB_DIR"

# ── Conda + CUDA ──
CONDA_ENV_NAME=${CONDA_ENV_NAME:-cxtgraph}
export CONDA_ENV_NAME
set +u  # conda activation scripts reference unbound vars (PS1, _CE_CONDA) -> set -u would kill us silently
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"
set -u
export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
CUDA_TARGET_LIB=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib
CUDA_LIB=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64
# A shell that switches from cxtgraph to a cloned environment can retain the
# old environment's lib directory. Mixing both prefixes can crash Ray workers
# in ld.so before Python has a chance to report an exception.
SYSTEM_LD_LIBRARY_PATH=""
IFS=: read -ra LIBRARY_PATH_ENTRIES <<< "${LD_LIBRARY_PATH:-}"
for library_path_entry in "${LIBRARY_PATH_ENTRIES[@]}"; do
  case "$library_path_entry" in
    ""|*/miniconda3/envs/*|"$CUDA_TARGET_LIB"|"$CUDA_LIB") ;;
    *) SYSTEM_LD_LIBRARY_PATH="${SYSTEM_LD_LIBRARY_PATH:+$SYSTEM_LD_LIBRARY_PATH:}$library_path_entry" ;;
  esac
done
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CUDA_TARGET_LIB}:${CUDA_LIB}${SYSTEM_LD_LIBRARY_PATH:+:$SYSTEM_LD_LIBRARY_PATH}
# Vista's aarch64 glibc can hit its pthread_create/dlopen TLS race when a
# threaded Ray worker imports PyTorch and its native dependencies lazily.
# Load both OpenMP and PyTorch's global dependency bundle before Ray creates
# worker threads so importing torch does not mutate the TLS layout afterward.
LIBGOMP_PATH=${CONDA_PREFIX}/lib/libgomp.so.1
if [ ! -f "$LIBGOMP_PATH" ]; then
  LIBGOMP_PATH=$(gcc -print-file-name=libgomp.so.1)
fi
if [ ! -f "$LIBGOMP_PATH" ]; then
  echo "ERROR: could not locate libgomp.so.1 for the aarch64 TLS preload workaround"
  exit 1
fi
export LD_PRELOAD=$LIBGOMP_PATH
TORCH_GLOBAL_DEPS_PATH=$(python -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("torch"); print(pathlib.Path(spec.origin).parent / "lib" / "libtorch_global_deps.so") if spec and spec.origin else print("")')
if [ ! -f "$TORCH_GLOBAL_DEPS_PATH" ]; then
  echo "ERROR: could not locate torch/lib/libtorch_global_deps.so for the aarch64 TLS preload workaround"
  exit 1
fi
export LD_PRELOAD=${LD_PRELOAD}:$TORCH_GLOBAL_DEPS_PATH
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
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# Per-trajectory sandbox workdir root (scratch is fastest on Vista)
export SAB_WORKDIR_ROOT=${SAB_WORKDIR_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/sab_workdirs}
mkdir -p "$SAB_WORKDIR_ROOT"
export DISCOVERYBENCH_WORKDIR_ROOT=${DISCOVERYBENCH_WORKDIR_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/discoverybench_workdirs}
export DISCOVERYBENCH_RESULTS_DIR=${DISCOVERYBENCH_RESULTS_DIR:-${SCRATCH:-/scratch/09281/chc_1996}/discoverybench_results/${SLURM_JOB_ID:-local}}
export DISCOVERYBENCH_REAL_EVAL=${DISCOVERYBENCH_REAL_EVAL:-0}
export DISCOVERYBENCH_JUDGE_MODEL=${DISCOVERYBENCH_JUDGE_MODEL:-gpt-5-nano}
mkdir -p "$DISCOVERYBENCH_WORKDIR_ROOT" "$DISCOVERYBENCH_RESULTS_DIR"

# ── Node info ──
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-8}

if [ "$NUM_NODES" -ne "$EXPECTED_NUM_NODES" ]; then
  echo "Expected $EXPECTED_NUM_NODES nodes, got $NUM_NODES"
  exit 1
fi

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$NUM_NODES}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}
SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:-1}
SAB_TRAIN_MAX_SAMPLES=${SAB_TRAIN_MAX_SAMPLES:-$TRAIN_BATCH_SIZE}
SAB_PROMPT_LENGTH=${SAB_PROMPT_LENGTH:-16384}
SAB_RESPONSE_LENGTH=${SAB_RESPONSE_LENGTH:-2048}
SAB_MAX_TOKEN_LEN_PER_GPU=${SAB_MAX_TOKEN_LEN_PER_GPU:-18432}
SAB_ROLLOUT_QUANTIZATION=${SAB_ROLLOUT_QUANTIZATION:-fp8}
SAB_ROLLOUT_GPU_MEMORY_UTILIZATION=${SAB_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
SAB_VAL_MAX_TURN=${SAB_VAL_MAX_TURN:-4}
SAB_TURN_MAX_NEW_TOKENS=${SAB_TURN_MAX_NEW_TOKENS:-512}
SAB_DATA_SEED=${SAB_DATA_SEED:-42}
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
if [ -n "$CODE_BENCHMARK_DATA_FILE" ]; then
  SAB_DATA_FILE=$CODE_BENCHMARK_DATA_FILE
fi
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
  probe_msg="wandb enabled (key length=${#WANDB_API_KEY}, source=$WANDB_ENV_SOURCE, dir=$WANDB_DIR)"
elif [ -f "$HOME/.netrc" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  probe_msg="wandb enabled via $HOME/.netrc (dir=$WANDB_DIR)"
else
  TRAINER_LOGGER='["console"]'
  probe_msg="WARNING: no WANDB_API_KEY or $HOME/.netrc found; eval will only log to console"
fi

TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_${SAB_METHOD}_sab_30b_instruct_${NUM_NODES}n_${SAB_RUN_TAG}_${TS}}

TRAINER_DEBUG_OVERRIDES=()
if [ "$SAB_DUMP_VALIDATION" = "1" ]; then
  SAB_VALIDATION_DATA_DIR=${SAB_VALIDATION_DATA_DIR:-${SCRATCH:-/scratch/09281/chc_1996}/sab_validation_generations/$EXPERIMENT_NAME}
  mkdir -p "$SAB_VALIDATION_DATA_DIR"
  TRAINER_DEBUG_OVERRIDES+=("trainer.validation_data_dir=$SAB_VALIDATION_DATA_DIR")
fi

ROLLOUT_QUANTIZATION_OVERRIDES=()
if [ "$SAB_ROLLOUT_QUANTIZATION" != "none" ]; then
  ROLLOUT_QUANTIZATION_OVERRIDES+=("+actor_rollout_ref.rollout.quantization=$SAB_ROLLOUT_QUANTIZATION")
fi

export EXPERIMENT_NAME
export WANDB_RUN_ID=${WANDB_RUN_ID:-$EXPERIMENT_NAME}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export WANDB_RESUME=${WANDB_RESUME:-allow}

wandb_status() {
  local phase="$1"
  local rc="${2:-0}"
  if [[ "$TRAINER_LOGGER" != *wandb* ]]; then
    return 0
  fi
  WANDB_PHASE="$phase" WANDB_STATUS_RC="$rc" python - <<'PY' || true
import os
import time

try:
    import wandb

    run = wandb.init(
        project="context-graph",
        name=os.environ.get("WANDB_NAME") or os.environ.get("EXPERIMENT_NAME"),
        id=os.environ.get("WANDB_RUN_ID"),
        resume=os.environ.get("WANDB_RESUME", "allow"),
        dir=os.environ.get("WANDB_DIR"),
        reinit=True,
    )
    phase = os.environ.get("WANDB_PHASE", "unknown")
    rc = int(os.environ.get("WANDB_STATUS_RC", "0"))
    wandb.log({
        "sab_script/heartbeat": 1,
        "sab_script/rc": rc,
        "sab_script/time": time.time(),
    }, step=0)
    run.summary["sab_script/phase"] = phase
    run.summary["sab_script/rc"] = rc
    run.summary["sab_script/local_self_log"] = os.environ.get("SELF_LOG_PATH", "")
    log_path = os.environ.get("SELF_LOG_PATH", "")
    if log_path and os.path.exists(log_path):
        wandb.save(log_path, policy="now")
    run.finish()
except Exception as exc:
    print(f"WARN: wandb status log failed: {exc}")
PY
}

probe() { printf '+++ [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

echo "=============================================================="
echo "  ZERO-SHOT EVAL: $SAB_METHOD_LABEL ($SAB_WORKFLOW) on $CODE_BENCHMARK_LABEL ($MODEL_PATH, $NUM_NODES nodes, ${SAB_RUN_TAG^^}, val_only=True)"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Workers: ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  Conda env:      $CONDA_ENV_NAME"
echo "  Conda prefix:   $CONDA_PREFIX"
echo "  Experiment:     $EXPERIMENT_NAME"
echo "  Sandbox workdir root: $SAB_WORKDIR_ROOT"
echo "  Logger: ${probe_msg}"
echo "  Batch sizes:    train=$TRAIN_BATCH_SIZE ppo_mini=$PPO_MINI_BATCH_SIZE"
echo "  Sample caps:    val=$SAB_VAL_MAX_SAMPLES train=$SAB_TRAIN_MAX_SAMPLES seed=$SAB_DATA_SEED"
echo "  Train samples:  $SAB_TRAIN_MAX_SAMPLES (trainer init only; val_only=True)"
echo "  Length caps:    prompt=$SAB_PROMPT_LENGTH response=$SAB_RESPONSE_LENGTH max_tokens_per_gpu=$SAB_MAX_TOKEN_LEN_PER_GPU"
echo "  Turn caps:      val_max_turn=$SAB_VAL_MAX_TURN turn_max_new_tokens=$SAB_TURN_MAX_NEW_TOKENS"
echo "  Qwen thinking:  $QWEN_ENABLE_THINKING"
echo "  Evaluation:     real=$SAB_REAL_EVAL dump_validation=$SAB_DUMP_VALIDATION ${SAB_VALIDATION_DATA_DIR:+dir=$SAB_VALIDATION_DATA_DIR}"
if [ "$CODE_BENCHMARK_PROFILE" = "discoverybench" ]; then
  echo "  Discovery HMS:  real=$DISCOVERYBENCH_REAL_EVAL judge=$DISCOVERYBENCH_JUDGE_MODEL results=$DISCOVERYBENCH_RESULTS_DIR"
fi
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: data parquets must exist ──
probe "checking $CODE_BENCHMARK_LABEL artefacts"
VAL_PARQUET="$PROJECT_ROOT/$SAB_DATA_FILE"
if [ ! -f "$VAL_PARQUET" ]; then
  echo "ERROR: missing $VAL_PARQUET"
  if [ -n "$CODE_BENCHMARK_PREPARE_HINT" ]; then
    echo "       $CODE_BENCHMARK_PREPARE_HINT"
  else
    echo "       Run: python scripts/make_sab_data.py --csv data/ScienceAgentBench.csv \\"
    echo "                 --benchmark-dir data/sab_benchmark --out-dir data"
  fi
  exit 1
fi
# verl wants a train_files path too even with val_only=True; reuse the same file.
TRAIN_PARQUET="$VAL_PARQUET"
probe "SAB parquet: $VAL_PARQUET"

if [ "$CODE_BENCHMARK_PROFILE" = "sab" ] && [ "$SAB_REAL_EVAL" = "1" ]; then
  if [ ! -f "$PROJECT_ROOT/gpt4_visual_judge.py" ]; then
    echo "ERROR: missing $PROJECT_ROOT/gpt4_visual_judge.py"
    echo "       Visual SAB evaluators cannot run without this helper."
    exit 1
  fi
  if [ -n "${OPENAI_API_KEY:-}" ] && [ "${OPENAI_API_KEY:-}" != "dummy" ]; then
    probe "real evaluator visual judge: OpenAI credentials loaded from $OPENAI_ENV_SOURCE"
  elif { [ -n "${AZURE_OPENAI_KEY:-}" ] || [ -n "${AZURE_OPENAI_API_KEY:-}" ]; } && [ -n "${AZURE_OPENAI_API_VERSION:-}" ] && [ -n "${AZURE_OPENAI_ENDPOINT:-}" ] && [ -n "${AZURE_OPENAI_DEPLOYMENT_NAME:-}" ]; then
    probe "real evaluator visual judge: Azure OpenAI credentials loaded from $OPENAI_ENV_SOURCE"
  else
    echo "ERROR: SAB_REAL_EVAL=1 requires a real OPENAI_API_KEY or the complete Azure OpenAI credential set"
    echo "       Required Azure variables: AZURE_OPENAI_KEY, AZURE_OPENAI_API_VERSION,"
    echo "       AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME."
    echo "       Store exported variables in \$WORK/.openai_env with mode 600."
    exit 1
  fi
fi

if [ "$CODE_BENCHMARK_PROFILE" = "discoverybench" ] && [ "$DISCOVERYBENCH_REAL_EVAL" = "1" ]; then
  if [ -n "${OPENAI_API_KEY:-}" ] && [ "${OPENAI_API_KEY:-}" != "dummy" ]; then
    probe "DiscoveryBench HMS judge: OpenAI credentials loaded from $OPENAI_ENV_SOURCE"
  elif { [ -n "${AZURE_OPENAI_KEY:-}" ] || [ -n "${AZURE_OPENAI_API_KEY:-}" ]; } && [ -n "${AZURE_OPENAI_API_VERSION:-}" ] && [ -n "${AZURE_OPENAI_ENDPOINT:-}" ] && [ -n "${AZURE_OPENAI_DEPLOYMENT_NAME:-}" ]; then
    probe "DiscoveryBench HMS judge: Azure OpenAI credentials loaded from $OPENAI_ENV_SOURCE"
  else
    echo "ERROR: DISCOVERYBENCH_REAL_EVAL=1 requires OPENAI_API_KEY or the complete Azure OpenAI credential set"
    exit 1
  fi
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

# Fail before starting a multi-node Ray cluster when the installed Transformers
# build cannot recognize a newly released model architecture.
probe "checking Transformers support for $MODEL_PATH"
python scripts/check_hf_model_support.py "$MODEL_PATH"
probe "Transformers model support check passed"
probe "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
probe "LD_PRELOAD=$LD_PRELOAD"

# ── Topology: NODE0 = Ray head + trainer rank 0; NODE1-4 = Ray workers ──
TRAINER_HEAD_NODE=${NODELIST[0]}
TRAINER_HEAD_IP=$(getent hosts "$TRAINER_HEAD_NODE" | awk '{print $1}')
echo "  Trainer Ray head:      $TRAINER_HEAD_NODE ($TRAINER_HEAD_IP)"
echo "  Trainer workers:       ${NODELIST[@]:1}"

# ── Stale Ray cleanup on all nodes ──
probe "ray stop sweep across $NUM_NODES nodes"
RAY_STOP_SRUN_TIMEOUT_SECONDS=${RAY_STOP_SRUN_TIMEOUT_SECONDS:-60}
for node in "${NODELIST[@]}"; do
  probe "ray stop start node=$node timeout=${RAY_STOP_SRUN_TIMEOUT_SECONDS}s"
  if timeout --signal=TERM --kill-after=10s "${RAY_STOP_SRUN_TIMEOUT_SECONDS}s" \
    srun --overlap --kill-on-bad-exit=1 --nodes=1 --ntasks=1 -w "$node" bash -c '
      source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
      conda activate '"$CONDA_ENV_NAME"'
      export LD_LIBRARY_PATH='"$LD_LIBRARY_PATH"'
      export LD_PRELOAD='"$LD_PRELOAD"'
      timeout --signal=TERM --kill-after=5s 30s ray stop -f >/dev/null 2>&1 || true
    '
  then
    probe "ray stop done node=$node"
  else
    ray_stop_rc=$?
    probe "WARNING: ray stop skipped node=$node rc=$ray_stop_rc after timeout/error"
  fi
done
sleep 5
probe "ray stop sweep done"

# ── Sanity imports ──
probe "python sanity imports"
python -c "import torch; print('torch:', torch.__version__, 'cuda available:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"
if [ "$CODE_BENCHMARK_PROFILE" = "discoverybench" ]; then
  python -c "from envs.discoverybench_env import DiscoveryBenchEnv; from envs.discoverybench_eval import score_hypothesis; print('DiscoveryBench env OK')"
else
  python -c "from envs.scienceagent_sandbox import CodeSandbox; from envs.scienceagent_env import ScienceAgentEnv; print('SAB env OK')"
fi
probe "sanity imports done"

# ── Ray head on NODELIST[0] ──
probe "starting Ray head on $TRAINER_HEAD_NODE"
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" bash -c '
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate '"$CONDA_ENV_NAME"'
  export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  hash -r
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
  export LD_LIBRARY_PATH='"$LD_LIBRARY_PATH"'
  export LD_PRELOAD='"$LD_PRELOAD"'
  export HF_HOME='"$HF_HOME"'
  export HF_HUB_CACHE='"$HF_HUB_CACHE"'
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export SAB_WORKDIR_ROOT='"$SAB_WORKDIR_ROOT"'
  export SAB_REAL_EVAL='"$SAB_REAL_EVAL"'
  export SAB_EXPOSE_EVAL_CONTRACT='"$SAB_EXPOSE_EVAL_CONTRACT"'
  export SAB_INTERACTIVE_EVAL_FEEDBACK='"$SAB_INTERACTIVE_EVAL_FEEDBACK"'
  export DISCOVERYBENCH_WORKDIR_ROOT='"$DISCOVERYBENCH_WORKDIR_ROOT"'
  export DISCOVERYBENCH_RESULTS_DIR='"$DISCOVERYBENCH_RESULTS_DIR"'
  export DISCOVERYBENCH_REAL_EVAL='"$DISCOVERYBENCH_REAL_EVAL"'
  export DISCOVERYBENCH_JUDGE_MODEL='"$DISCOVERYBENCH_JUDGE_MODEL"'
  export QWEN_ENABLE_THINKING='"$QWEN_ENABLE_THINKING"'
  ray start --head --node-ip-address='"$TRAINER_HEAD_IP"' --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
' &
RAY_HEAD_PID=$!
sleep 20
probe "Ray head sleep done; launching $((NUM_NODES - 1)) trainer workers"

# ── Ray workers on NODELIST[1..N-1] ──
WORKER_PIDS=()
for i in $(seq 1 $((NUM_NODES - 1))); do
  WORKER_NODE=${NODELIST[$i]}
  srun --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c '
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate '"$CONDA_ENV_NAME"'
    export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    hash -r
    export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
    export LD_LIBRARY_PATH='"$LD_LIBRARY_PATH"'
    export LD_PRELOAD='"$LD_PRELOAD"'
    export HF_HOME='"$HF_HOME"'
    export HF_HUB_CACHE='"$HF_HUB_CACHE"'
    export FLASHINFER_WORKSPACE_BASE=/tmp
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export SAB_WORKDIR_ROOT='"$SAB_WORKDIR_ROOT"'
    export SAB_REAL_EVAL='"$SAB_REAL_EVAL"'
    export SAB_EXPOSE_EVAL_CONTRACT='"$SAB_EXPOSE_EVAL_CONTRACT"'
    export SAB_INTERACTIVE_EVAL_FEEDBACK='"$SAB_INTERACTIVE_EVAL_FEEDBACK"'
    export DISCOVERYBENCH_WORKDIR_ROOT='"$DISCOVERYBENCH_WORKDIR_ROOT"'
    export DISCOVERYBENCH_RESULTS_DIR='"$DISCOVERYBENCH_RESULTS_DIR"'
    export DISCOVERYBENCH_REAL_EVAL='"$DISCOVERYBENCH_REAL_EVAL"'
    export DISCOVERYBENCH_JUDGE_MODEL='"$DISCOVERYBENCH_JUDGE_MODEL"'
  export QWEN_ENABLE_THINKING='"$QWEN_ENABLE_THINKING"'
    ray start --address='"${TRAINER_HEAD_IP}:6379"' --num-cpus=70 --num-gpus=1 --block
  ' &
  WORKER_PIDS+=("$!")
  sleep 5
done
sleep 20
probe "all $((NUM_NODES - 1)) trainer workers launched, cluster settling"

cleanup() {
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
echo "  Launching $SAB_METHOD_LABEL ($SAB_WORKFLOW) ZERO-SHOT eval ($NUM_NODES nodes ${SAB_RUN_TAG^^}, $CODE_BENCHMARK_LABEL test cap=$SAB_VAL_MAX_SAMPLES)"
echo "  default_agent_loop=$SAB_AGENT_LOOP  workflow=$SAB_WORKFLOW  process_reward=$SAB_PROCESS_REWARD"
echo "  vLLM gpu_memory_utilization=$SAB_ROLLOUT_GPU_MEMORY_UTILIZATION quantization=$SAB_ROLLOUT_QUANTIZATION + FSDP CPU offload"
echo "  val_only=True (one val pass on $SAB_DATA_FILE then exit; no training)"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~10-15 min)"
wandb_status trainer_launch 0
RAY_LOG_MARKER=$(mktemp /tmp/qwen3-30b-ray-log-marker.XXXXXX)

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" --chdir="$PROJECT_ROOT" \
  --export=ALL,SAB_WORKDIR_ROOT="$SAB_WORKDIR_ROOT",SAB_REAL_EVAL="$SAB_REAL_EVAL",SAB_EXPOSE_EVAL_CONTRACT="$SAB_EXPOSE_EVAL_CONTRACT",SAB_INTERACTIVE_EVAL_FEEDBACK="$SAB_INTERACTIVE_EVAL_FEEDBACK",DISCOVERYBENCH_WORKDIR_ROOT="$DISCOVERYBENCH_WORKDIR_ROOT",DISCOVERYBENCH_RESULTS_DIR="$DISCOVERYBENCH_RESULTS_DIR",DISCOVERYBENCH_REAL_EVAL="$DISCOVERYBENCH_REAL_EVAL",DISCOVERYBENCH_JUDGE_MODEL="$DISCOVERYBENCH_JUDGE_MODEL",QWEN_ENABLE_THINKING="$QWEN_ENABLE_THINKING" \
  python -m "$CODE_BENCHMARK_TRAIN_MODULE" \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.005 \
  actor_rollout_ref.rollout.agent.default_agent_loop=$SAB_AGENT_LOOP \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  "${ROLLOUT_QUANTIZATION_OVERRIDES[@]}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=$SAB_ROLLOUT_GPU_MEMORY_UTILIZATION \
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
  data.seed=$SAB_DATA_SEED \
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
  trainer.logger="$TRAINER_LOGGER" \
  "${TRAINER_DEBUG_OVERRIDES[@]}"
RC=$?
set -e
if [ $RC -ne 0 ]; then
  RAY_LOG_DIR=/tmp/ray/session_latest/logs
  probe "collecting recent non-empty Ray worker logs from $RAY_LOG_DIR"
  if [ -d "$RAY_LOG_DIR" ]; then
    while IFS= read -r -d '' ray_log; do
      echo "----- RAY LOG: $ray_log -----"
      tail -n 160 "$ray_log" || true
    done < <(find "$RAY_LOG_DIR" -maxdepth 1 -type f -newer "$RAY_LOG_MARKER" -size +0c \( -name 'worker-*.err' -o -name 'python-core-worker-*.log' -o -name 'raylet.err' \) -print0)
  fi
fi
rm -f "$RAY_LOG_MARKER"
wandb_status trainer_exit "$RC"

echo "=============================================================="
if [ $RC -eq 0 ]; then
  echo "  EVAL RUN COMPLETED (exit 0)"
else
  echo "  EVAL RUN FAILED (exit $RC); check above for first error"
fi
echo "  Finished: $(date)"
echo "=============================================================="

exit $RC
