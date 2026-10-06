#!/bin/bash
#SBATCH -J train-bc-8b-cg-paperfaithful-5n-48h
#SBATCH -o logs/train-bc-8b-cg-paperfaithful-5n-48h.%j.out
#SBATCH -e logs/train-bc-8b-cg-paperfaithful-5n-48h.%j.err
#SBATCH -p gh
#SBATCH -N 5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 48:00:00
#SBATCH -A AST24021

# ─────────────────────────────────────────────────────────────────────
# 24-hour 4-NODE training (v2: post-fix) for BrowseComp-Plus ctxgraph 8B at 16K response.
#
# v2 reward changes (vs 10h baseline):
#   1. concise_main process penalty: -1 fixed → continuous (-0.5 floor, -1 at mean, -2 cap)
#      ↳ stronger pressure to shorten long non-op main turns
#   2. graph_shaping semi-de-gated: failed-task path now gives 0.3*(usage + 0.5*structural)
#      ↳ policy can learn graph ops have value even on failed trajectories
#   3. lambda_cost: 0.002 → 0.02 (10×) — balances the de-gating to prevent reward hacking
#
# 24h: total_training_steps=30, val + ckpt every 10 steps (6 ckpts: 10/20/.../60).
# Step time in 10h baseline ~17-22 min/step → 60 steps ≈ 17-22h.
# 10-hour 4-NODE training (full 30-step fit) for BrowseComp-Plus ctxgraph 8B at 16K response.
# Topology: NODELIST[0] = dedicated search server (no Ray), NODELIST[1]
# = Ray head + trainer rank 0, NODELIST[2,3] = Ray workers. FSDP across 3 GPUs.
# Production training (30 steps, val every 10 steps, ckpt every 10 steps):
#   - search_server.py loads HF datasets (Tevatron/browsecomp-plus-corpus +
#     precomputed embeddings from miaolu3/browsecomp-plus)
#   - trainer hits OpenAI judge per rollout completion (gpt-5-nano default,
#     override via JUDGE_MODEL env var; cheaper smoke: JUDGE_MODEL=gpt-4o-mini)
#   - ctxgraph agent loop emits graph ops over BC's multi-page browsing tasks
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
if [ "${BC_REQUIRE_WANDB:-0}" = "1" ]; then
  if [ "${BC_DISABLE_WANDB:-0}" = "1" ]; then
    echo "ERROR: BC_REQUIRE_WANDB=1 conflicts with BC_DISABLE_WANDB=1"
    exit 1
  fi
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: BC_REQUIRE_WANDB=1 but WANDB_API_KEY is not set"
    echo "       Put 'export WANDB_API_KEY=...' in \$WORK/.wandb_env"
    exit 1
  fi
  unset WANDB_DISABLED
  export WANDB_MODE=online
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
conda activate ${TRAIN_CONDA_ENV:-cxtgraph}
export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r

export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
IFS=: read -ra LD_LIBRARY_ENTRIES <<< "${LD_LIBRARY_PATH:-}"
CLEAN_LD_LIBRARY_PATH=
for LD_LIBRARY_ENTRY in "${LD_LIBRARY_ENTRIES[@]}"; do
  if [ -z "$LD_LIBRARY_ENTRY" ] || [[ "$LD_LIBRARY_ENTRY" == *"/envs/graphtrl/lib"* ]]; then
    continue
  fi
  CLEAN_LD_LIBRARY_PATH="${CLEAN_LD_LIBRARY_PATH:+$CLEAN_LD_LIBRARY_PATH:}$LD_LIBRARY_ENTRY"
done
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/targets/sbsa-linux/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64${CLEAN_LD_LIBRARY_PATH:+:$CLEAN_LD_LIBRARY_PATH}
unset LD_LIBRARY_ENTRIES LD_LIBRARY_ENTRY CLEAN_LD_LIBRARY_PATH
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
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# ── Node info ──
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ -n "${BC_ACTIVE_NODE_COUNT:-}" ]; then
  [[ "$BC_ACTIVE_NODE_COUNT" =~ ^[1-9][0-9]*$ ]] && [ "$BC_ACTIVE_NODE_COUNT" -ge 3 ] && [ "$BC_ACTIVE_NODE_COUNT" -le "${#NODELIST[@]}" ] || { echo "Invalid BC_ACTIVE_NODE_COUNT: $BC_ACTIVE_NODE_COUNT" >&2; exit 2; }
  # Restrict every service, Ray sweep and trainer to this allocation prefix.
  # Keep Slurm's allocation variables intact; the remaining nodes are untouched.
  NODELIST=("${NODELIST[@]:0:$BC_ACTIVE_NODE_COUNT}")
fi
NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
NUM_NODES=${#NODELIST[@]}
EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-5}

if [ "$NUM_NODES" -ne "$EXPECTED_NUM_NODES" ]; then
  echo "Expected $EXPECTED_NUM_NODES nodes, got $NUM_NODES"
  exit 1
fi

if [ "${BC_DISABLE_WANDB:-0}" = "1" ]; then
  unset WANDB_API_KEY
  TRAINER_LOGGER='["console"]'
  probe_msg="wandb disabled by BC_DISABLE_WANDB=1"
elif [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
  probe_msg="wandb enabled (key length=${#WANDB_API_KEY})"
else
  TRAINER_LOGGER='["console"]'
  probe_msg="WARNING: no WANDB_API_KEY in env — training will only log to console"
fi

TS=$(date +%Y%m%d_%H%M%S)
RUN_TAG=${RUN_TAG:-paperfaithful_5n_48h}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"train_ctxgraph_bc_8b_${RUN_TAG}_${TS}"}
SAVE_ROLLOUT_DATA=${SAVE_ROLLOUT_DATA:-0}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-}
if [ "$SAVE_ROLLOUT_DATA" = "1" ] && [ -z "$ROLLOUT_DATA_DIR" ]; then
  ROLLOUT_DATA_ROOT=${ROLLOUT_DATA_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/context-graph-rollouts}
  ROLLOUT_DATA_DIR="$ROLLOUT_DATA_ROOT/$EXPERIMENT_NAME"
fi
ROLLOUT_DATA_ARGS=()
if [ -n "$ROLLOUT_DATA_DIR" ]; then
  mkdir -p "$ROLLOUT_DATA_DIR"
  ROLLOUT_DATA_ARGS+=("trainer.rollout_data_dir=$ROLLOUT_DATA_DIR")
fi
VALIDATION_DATA_ARGS=()
if [ -n "$VALIDATION_DATA_DIR" ]; then
  mkdir -p "$VALIDATION_DATA_DIR"
  VALIDATION_DATA_ARGS+=("trainer.validation_data_dir=$VALIDATION_DATA_DIR")
fi
DATA_SEED_ARGS=()
if [ -n "${DATA_SEED:-}" ]; then
  DATA_SEED_ARGS+=("data.seed=$DATA_SEED")
fi
TRAIN_DATA_FILE=${TRAIN_DATA_FILE:-data/bc_train.parquet}
VAL_DATA_FILE=${VAL_DATA_FILE:-data/bc_test.parquet}
DATASET_LABEL=${DATASET_LABEL:-BrowseComp-Plus}
LOCAL_SEARCH_CORPUS=${LOCAL_SEARCH_CORPUS:-}
LOCAL_SEARCH_EMBEDDINGS=${LOCAL_SEARCH_EMBEDDINGS:-}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
TRAINER_VAL_ONLY=${TRAINER_VAL_ONLY:-False}
BC_CTXGRAPH_PROTOCOL=${BC_CTXGRAPH_PROTOCOL:-legacy}
BC_CONTROLLER_ACTION_POLICY=${BC_CONTROLLER_ACTION_POLICY:-structural}
PROMPT_LENGTH=${PROMPT_LENGTH:-8192}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-32768}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-40960}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_FREQ=${SAVE_FREQ:-10}
BC_SEARCH_TIMEOUT_SECONDS=${BC_SEARCH_TIMEOUT_SECONDS:-600}
SESSION_TIMEOUT=${SESSION_TIMEOUT:-3600}
MAX_TURN=${MAX_TURN:-100}
MAX_SESSION=${MAX_SESSION:-10}
VAL_MAX_SESSION=${VAL_MAX_SESSION:-10}
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-2048}
ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-True}
FINAL_ANSWER_RESERVE=${FINAL_ANSWER_RESERVE:-2048}
FINAL_ANSWER_SAFETY_MARGIN=${FINAL_ANSWER_SAFETY_MARGIN:-64}
CONSOLIDATION_INTERVAL=${CONSOLIDATION_INTERVAL:-5}
AUTO_PRUNE_MAX_ACTIVE=${AUTO_PRUNE_MAX_ACTIVE:-12}
STRUCTURED_MEMORY_ENABLED=${STRUCTURED_MEMORY_ENABLED:-0}
STRUCTURED_MEMORY_REQUIRED=${STRUCTURED_MEMORY_REQUIRED:-$STRUCTURED_MEMORY_ENABLED}
STRUCTURED_MEMORY_GAP_INTERVAL=${STRUCTURED_MEMORY_GAP_INTERVAL:-8}
STRUCTURED_MEMORY_CONTEXT_BUDGET=${STRUCTURED_MEMORY_CONTEXT_BUDGET:-1024}
STRUCTURED_MEMORY_MAX_CONTEXT_FACTS=${STRUCTURED_MEMORY_MAX_CONTEXT_FACTS:-12}
STRUCTURED_MEMORY_MAX_FACTS_PER_OBSERVATION=${STRUCTURED_MEMORY_MAX_FACTS_PER_OBSERVATION:-8}
STRUCTURED_MEMORY_MAX_FACTS=${STRUCTURED_MEMORY_MAX_FACTS:-128}
STRUCTURED_MEMORY_EXTRACT_MAX_TOKENS=${STRUCTURED_MEMORY_EXTRACT_MAX_TOKENS:-768}
STRUCTURED_MEMORY_GAP_MAX_TOKENS=${STRUCTURED_MEMORY_GAP_MAX_TOKENS:-512}
STRUCTURED_MEMORY_PLAN_MAX_TOKENS=${STRUCTURED_MEMORY_PLAN_MAX_TOKENS:-512}
STRUCTURED_MEMORY_RELATION_CANDIDATES=${STRUCTURED_MEMORY_RELATION_CANDIDATES:-128}
STRUCTURED_MEMORY_CONTROLLER_RETRIES=${STRUCTURED_MEMORY_CONTROLLER_RETRIES:-2}
STRUCTURED_MEMORY_GAP_JITTER=${STRUCTURED_MEMORY_GAP_JITTER:-1}
STRUCTURED_MEMORY_STOP_ON_READY=${STRUCTURED_MEMORY_STOP_ON_READY:-1}
STRUCTURED_MEMORY_STEP_LIMIT=${STRUCTURED_MEMORY_STEP_LIMIT:-40}
TRAIN_LR=${TRAIN_LR:-2e-6}
USE_KL_LOSS=${USE_KL_LOSS:-True}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-foldgrpo}
POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla}
PROCESS_REWARD_SPEC=${PROCESS_REWARD_SPEC:-'[flat,scope,graph]'}

case "$BC_CTXGRAPH_PROTOCOL" in
  legacy|full_policy)
    BC_STRUCTURED_GRAPH_CONTROLLER=false
    BC_CONTROLLER_OWNED_TOOL_FORMATTING=false
    ;;
  controller)
    BC_STRUCTURED_GRAPH_CONTROLLER=true
    BC_CONTROLLER_OWNED_TOOL_FORMATTING=true
    ;;
  *)
    echo "ERROR: BC_CTXGRAPH_PROTOCOL must be full_policy, legacy, or controller; got $BC_CTXGRAPH_PROTOCOL"
    exit 1
    ;;
esac
case "$STRUCTURED_MEMORY_ENABLED" in
  0) STRUCTURED_MEMORY_ENABLED=false ;;
  1) STRUCTURED_MEMORY_ENABLED=true ;;
  *) echo "ERROR: STRUCTURED_MEMORY_ENABLED must be 0 or 1"; exit 1 ;;
esac
case "$STRUCTURED_MEMORY_REQUIRED" in
  0) STRUCTURED_MEMORY_REQUIRED=false ;;
  1) STRUCTURED_MEMORY_REQUIRED=true ;;
  *) echo "ERROR: STRUCTURED_MEMORY_REQUIRED must be 0 or 1"; exit 1 ;;
esac
case "$STRUCTURED_MEMORY_STOP_ON_READY" in
  0) STRUCTURED_MEMORY_STOP_ON_READY=false ;;
  1) STRUCTURED_MEMORY_STOP_ON_READY=true ;;
  *) echo "ERROR: STRUCTURED_MEMORY_STOP_ON_READY must be 0 or 1"; exit 1 ;;
esac
if [ "$STRUCTURED_MEMORY_ENABLED" = "true" ] && [ "$BC_CTXGRAPH_PROTOCOL" != "controller" ]; then
  echo "ERROR: structured memory requires BC_CTXGRAPH_PROTOCOL=controller"
  exit 1
fi
if [ "$STRUCTURED_MEMORY_ENABLED" = "true" ] && [ "${TRAINER_VAL_ONLY,,}" != "true" ]; then
  echo "ERROR: structured memory is inference-only; set TRAINER_VAL_ONLY=True"
  exit 1
fi
case "$BC_CONTROLLER_ACTION_POLICY" in
  balanced|structural) ;;
  *)
    echo "ERROR: BC_CONTROLLER_ACTION_POLICY must be balanced or structural; got $BC_CONTROLLER_ACTION_POLICY"
    exit 1
    ;;
esac
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.0005}
ALGORITHM_KL_COEF=${ALGORITHM_KL_COEF:-0.005}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts/$EXPERIMENT_NAME}
TRAINER_RESUME_MODE=${TRAINER_RESUME_MODE:-auto}
case "$TRAINER_RESUME_MODE" in
  auto|disable) ;;
  *)
    echo "ERROR: TRAINER_RESUME_MODE must be auto or disable; got $TRAINER_RESUME_MODE"
    exit 1
    ;;
esac
if [ "$LORA_RANK" -gt 0 ]; then
  MODEL_UPDATE_LABEL="LoRA(r=$LORA_RANK,alpha=$LORA_ALPHA,target=$LORA_TARGET_MODULES)"
else
  MODEL_UPDATE_LABEL="full-parameter"
fi

GRAPH_RPO_ARGS=()
if [ "$ADV_ESTIMATOR" = "graphrpo" ]; then
  if [ "$BC_CTXGRAPH_PROTOCOL" != "controller" ]; then
    echo "ERROR: GraphRPO requires BC_CTXGRAPH_PROTOCOL=controller"
    exit 1
  fi
  GRAPH_RPO_CREDIT_BACKEND=${GRAPH_RPO_CREDIT_BACKEND:-old_policy_counterfactual_qa}
  GRAPH_RPO_ALPHA=${GRAPH_RPO_ALPHA:-1.0}
  GRAPH_RPO_BETA=${GRAPH_RPO_BETA:-1.0}
  GRAPH_RPO_EPSILON=${GRAPH_RPO_EPSILON:-1e-6}
  GRAPH_RPO_DELTA_SCALE=${GRAPH_RPO_DELTA_SCALE:-1.0}
  GRAPH_RPO_DELTA_MAX=${GRAPH_RPO_DELTA_MAX:-1.0}
  GRAPH_RPO_OPERATION_COSTS=${GRAPH_RPO_OPERATION_COSTS:-'{merge:0.0,prune:0.0,add_edge:0.0,select:0.0}'}
  GRAPH_RPO_ARGS+=(
    "algorithm.graphrpo_alpha=$GRAPH_RPO_ALPHA"
    "algorithm.graphrpo_beta=$GRAPH_RPO_BETA"
    "algorithm.graphrpo_epsilon=$GRAPH_RPO_EPSILON"
    "+actor_rollout_ref.rollout.plugin.graph_rpo_credit_backend=$GRAPH_RPO_CREDIT_BACKEND"
    "+actor_rollout_ref.rollout.plugin.graph_rpo_delta_scale=$GRAPH_RPO_DELTA_SCALE"
    "+actor_rollout_ref.rollout.plugin.graph_rpo_delta_max=$GRAPH_RPO_DELTA_MAX"
    "+actor_rollout_ref.rollout.plugin.graph_rpo_operation_costs=$GRAPH_RPO_OPERATION_COSTS"
  )
  case "$GRAPH_RPO_CREDIT_BACKEND" in
    evidence)
      python scripts/prepare_graph_evidence_data.py --check "$TRAIN_DATA_FILE"
      GRAPH_RPO_ARGS+=(
        "algorithm.graphrpo_normalize_decision_tokens=True"
        "algorithm.rollout_correction.rollout_is=token"
        "algorithm.rollout_correction.rollout_is_threshold=2.0"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_scope_process_reward=False"
        "+actor_rollout_ref.rollout.plugin.graph_branch_history=True"
        "+actor_rollout_ref.rollout.plugin.graph_controller_temperature=${GRAPH_CONTROLLER_TEMPERATURE:-0.8}"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_duplicate_threshold=${GRAPH_RPO_DUPLICATE_THRESHOLD:-0.6}"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_duplicate_penalty=${GRAPH_RPO_DUPLICATE_PENALTY:-0.2}"
      )
      ;;
    old_policy_counterfactual_qa)
      GRAPH_RPO_COUNTERFACTUAL_SAMPLES=${GRAPH_RPO_COUNTERFACTUAL_SAMPLES:-2}
      GRAPH_RPO_COUNTERFACTUAL_MAX_NEW_TOKENS=${GRAPH_RPO_COUNTERFACTUAL_MAX_NEW_TOKENS:-512}
      GRAPH_RPO_COUNTERFACTUAL_TEMPERATURE=${GRAPH_RPO_COUNTERFACTUAL_TEMPERATURE:-$ROLLOUT_TEMPERATURE}
      GRAPH_RPO_COUNTERFACTUAL_TOP_P=${GRAPH_RPO_COUNTERFACTUAL_TOP_P:-1.0}
      GRAPH_RPO_COUNTERFACTUAL_SEED=${GRAPH_RPO_COUNTERFACTUAL_SEED:-42}
      GRAPH_RPO_COUNTERFACTUAL_ENABLE_THINKING=${GRAPH_RPO_COUNTERFACTUAL_ENABLE_THINKING:-False}
      GRAPH_RPO_ARGS+=(
        "+actor_rollout_ref.rollout.plugin.graph_rpo_counterfactual_samples=$GRAPH_RPO_COUNTERFACTUAL_SAMPLES"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_counterfactual_max_new_tokens=$GRAPH_RPO_COUNTERFACTUAL_MAX_NEW_TOKENS"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_counterfactual_temperature=$GRAPH_RPO_COUNTERFACTUAL_TEMPERATURE"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_counterfactual_top_p=$GRAPH_RPO_COUNTERFACTUAL_TOP_P"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_counterfactual_seed=$GRAPH_RPO_COUNTERFACTUAL_SEED"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_counterfactual_enable_thinking=$GRAPH_RPO_COUNTERFACTUAL_ENABLE_THINKING"
      )
      ;;
    reference_answer_likelihood|old_policy_answer_likelihood)
      if [ "$USE_KL_LOSS" != "True" ] && [ "$USE_KL_LOSS" != "true" ]; then
        if [ "$GRAPH_RPO_CREDIT_BACKEND" = "reference_answer_likelihood" ]; then
          echo "ERROR: reference-answer GraphRPO requires USE_KL_LOSS=True to initialize the frozen reference policy"
          exit 1
        fi
      fi
      GRAPH_RPO_REFERENCE_MAX_PROMPT_LENGTH=${GRAPH_RPO_REFERENCE_MAX_PROMPT_LENGTH:-$PROMPT_LENGTH}
      GRAPH_RPO_REFERENCE_MAX_ANSWER_LENGTH=${GRAPH_RPO_REFERENCE_MAX_ANSWER_LENGTH:-128}
      GRAPH_RPO_REFERENCE_ENABLE_THINKING=${GRAPH_RPO_REFERENCE_ENABLE_THINKING:-False}
      GRAPH_RPO_ARGS+=(
        "+actor_rollout_ref.rollout.plugin.graph_rpo_reference_max_prompt_length=$GRAPH_RPO_REFERENCE_MAX_PROMPT_LENGTH"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_reference_max_answer_length=$GRAPH_RPO_REFERENCE_MAX_ANSWER_LENGTH"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_reference_enable_thinking=$GRAPH_RPO_REFERENCE_ENABLE_THINKING"
      )
      ;;
    external_evaluator)
      if [ -z "${GRAPH_RPO_EVALUATOR_URL:-}" ]; then
        echo "ERROR: external-evaluator GraphRPO requires GRAPH_RPO_EVALUATOR_URL"
        exit 1
      fi
      GRAPH_RPO_VIEW_BUDGET=${GRAPH_RPO_VIEW_BUDGET:-2048}
      GRAPH_RPO_PROBABILITY_EPSILON=${GRAPH_RPO_PROBABILITY_EPSILON:-1e-4}
      GRAPH_RPO_CONFIDENCE_BOUND=${GRAPH_RPO_CONFIDENCE_BOUND:-8.0}
      GRAPH_RPO_SERIALIZATION_PENALTY=${GRAPH_RPO_SERIALIZATION_PENALTY:-0.0}
      GRAPH_RPO_ARGS+=(
        "+actor_rollout_ref.rollout.plugin.graph_rpo_evaluator_url=$GRAPH_RPO_EVALUATOR_URL"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_view_budget=$GRAPH_RPO_VIEW_BUDGET"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_probability_epsilon=$GRAPH_RPO_PROBABILITY_EPSILON"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_confidence_bound=$GRAPH_RPO_CONFIDENCE_BOUND"
        "+actor_rollout_ref.rollout.plugin.graph_rpo_serialization_penalty=$GRAPH_RPO_SERIALIZATION_PENALTY"
      )
      ;;
    *)
      echo "ERROR: unsupported GRAPH_RPO_CREDIT_BACKEND (expected evidence, old_policy_counterfactual_qa, reference_answer_likelihood, old_policy_answer_likelihood, or external_evaluator)"
      exit 1
      ;;
  esac
fi

# Qwen3-8B advertises 40,960 positions. Longer runs must override both the
# actor/reference HF config and vLLM's independently loaded HF config.
LONG_CONTEXT_ARGS=()
if [ "$CONTEXT_LENGTH" -gt 40960 ] && [ "${BC_APPLY_YARN:-1}" = 1 ]; then
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
echo "  TRAIN: ContextGraph + v5 on $DATASET_LABEL ($MODEL_PATH, $NUM_NODES nodes [1 search + $((NUM_NODES - 1)) trainer], $TOTAL_TRAINING_STEPS steps)"
echo "  Token budget: prompt=$PROMPT_LENGTH response=$RESPONSE_LENGTH active_context=$CONTEXT_LENGTH"
echo "  Job: ${SLURM_JOB_ID:-<idev>}   Head: $NODE0 ($NODE0_IP)"
echo "  Worker(s): ${NODELIST[@]:1}"
echo "  Trainer model:  $MODEL_PATH"
echo "  Model update: $MODEL_UPDATE_LABEL"
echo "  Embedder model: $EMBED_MODEL"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Logger: ${probe_msg}"
echo "  Rollout data: ${ROLLOUT_DATA_DIR:-disabled}"
echo "  Validation data: ${VALIDATION_DATA_DIR:-disabled}"
echo "  Started: $(date)"
echo "=============================================================="

# ── Pre-flight: BrowseComp data parquets + HF datasets must exist ──
probe "checking training/evaluation parquets"
case "$TRAIN_DATA_FILE" in
  /*) TRAIN_PARQUET="$TRAIN_DATA_FILE" ;;
  *) TRAIN_PARQUET="$PROJECT_ROOT/$TRAIN_DATA_FILE" ;;
esac
case "$VAL_DATA_FILE" in
  /*) VAL_PARQUET="$VAL_DATA_FILE" ;;
  *) VAL_PARQUET="$PROJECT_ROOT/$VAL_DATA_FILE" ;;
esac
for f in "$TRAIN_PARQUET" "$VAL_PARQUET"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    exit 1
  fi
done
probe "data parquets ok: train=$TRAIN_DATA_FILE val=$VAL_DATA_FILE"

RESUME_ARGS=(trainer.resume_mode="$TRAINER_RESUME_MODE")
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
  probe "will load latest checkpoint $RESUME_PATH"
fi
CORPUS_DATASET="Tevatron/browsecomp-plus-corpus"
CORPUS_EMBEDDING_DATASET="miaolu3/browsecomp-plus"
SEARCH_SERVER_ARGS=(--model "$EMBED_MODEL" --port 18999)
if [ -n "$LOCAL_SEARCH_CORPUS" ] || [ -n "$LOCAL_SEARCH_EMBEDDINGS" ]; then
  if [ -z "$LOCAL_SEARCH_CORPUS" ] || [ -z "$LOCAL_SEARCH_EMBEDDINGS" ]; then
    echo "ERROR: LOCAL_SEARCH_CORPUS and LOCAL_SEARCH_EMBEDDINGS must be set together"
    exit 1
  fi
  for f in "$LOCAL_SEARCH_CORPUS" "$LOCAL_SEARCH_EMBEDDINGS"; do
    if [ ! -s "$f" ]; then
      echo "ERROR: missing or empty local retrieval artifact: $f"
      exit 1
    fi
  done
  SEARCH_SERVER_ARGS+=(--local-corpus "$LOCAL_SEARCH_CORPUS" --local-embeddings "$LOCAL_SEARCH_EMBEDDINGS")
  SEARCH_SOURCE="local corpus $LOCAL_SEARCH_CORPUS"
else
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
  SEARCH_SERVER_ARGS+=(--corpus "$CORPUS_DATASET" --corpus-embedding-dataset "$CORPUS_EMBEDDING_DATASET")
  SEARCH_SOURCE="Hugging Face corpus $CORPUS_DATASET"
fi
printf -v SEARCH_SERVER_ARGS_Q '%q ' "${SEARCH_SERVER_ARGS[@]}"
probe "retrieval artifacts ok: $SEARCH_SOURCE"

# ── Pre-flight: 8B + embedder weights must be present (offline) ──
probe "checking model caches"
if [ -d "$MODEL_PATH" ]; then
  TRAINER_CACHE_DIR="$MODEL_PATH"
  if [ ! -s "$TRAINER_CACHE_DIR/config.json" ]; then
    echo "ERROR: local MODEL_PATH is missing config.json: $TRAINER_CACHE_DIR"
    exit 1
  fi
  if ! find -L "$TRAINER_CACHE_DIR" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -size +0c -print -quit | grep -q .; then
    echo "ERROR: local MODEL_PATH has no non-empty Hugging Face weight files: $TRAINER_CACHE_DIR"
    exit 1
  fi
else
  TRAINER_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
fi
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
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,$SEARCH_NODE,$SEARCH_NODE_IP,$TRAINER_HEAD_NODE,$TRAINER_HEAD_IP"
export no_proxy="$NO_PROXY"
echo "  Dedicated search node: $SEARCH_NODE ($SEARCH_NODE_IP)"
echo "  Trainer Ray head:      $TRAINER_HEAD_NODE ($TRAINER_HEAD_IP)"
echo "  Trainer workers:       ${NODELIST[@]:2}"

# ── Start envs/search_server.py on dedicated SEARCH_NODE ──
probe "starting envs/search_server.py with $EMBED_MODEL and $SEARCH_SOURCE on dedicated $SEARCH_NODE:18999"
mkdir -p "$PROJECT_ROOT/logs"
SEARCH_LOG="$PROJECT_ROOT/logs/search-${SLURM_JOB_ID:-idev}-${RUN_TAG}-ctxgraph.log"
probe "search server log: $SEARCH_LOG"
srun --overlap --nodes=1 --ntasks=1 -w "$SEARCH_NODE" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate ${SEARCH_CONDA_ENV:-cxtgraph}
  if [ ${SEARCH_CLEAR_LD_PRELOAD:-0} = 1 ]; then unset LD_PRELOAD; fi
  cd $PROJECT_ROOT
  export PYTHONPATH=$PROJECT_ROOT:\${PYTHONPATH:-}
  export HF_HOME=$HF_HOME
  export HF_HUB_CACHE=$HF_HUB_CACHE
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export PYTHONUNBUFFERED=1
  export NUM_GPUS=1
  export MAX_BATCH_SIZE=128
  unset LOCAL_CORPUS_PARQUET LOCAL_EMBEDDINGS_PKL
  exec python -u envs/search_server.py $SEARCH_SERVER_ARGS_Q
" >"$SEARCH_LOG" 2>&1 &
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
    tail -80 "$SEARCH_LOG" || true
    exit 1
  fi
  sleep 1
done
if [ "$HEALTH_OK" != "1" ]; then
  echo "ERROR: search server did not become healthy within ${BC_SEARCH_TIMEOUT_SECONDS}s. Last 80 lines:"
  tail -80 "$SEARCH_LOG" || true
  kill "$SEARCH_PID" 2>/dev/null || true
  exit 1
fi

probe "waiting for search server /search probe (up to ${BC_SEARCH_TIMEOUT_SECONDS}s)"
SEARCH_OK=0
for _ in $(seq 1 "$BC_SEARCH_TIMEOUT_SECONDS"); do
  if curl --noproxy '*' -fsS -X POST -H 'Content-Type: application/json' \
      -d '{"query":"Eiffel Tower","k":1}' \
      "http://${SEARCH_NODE_IP}:18999/search" >/dev/null 2>&1; then
    SEARCH_OK=1
    break
  fi
  if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
    echo "ERROR: search server exited before /search probe succeeded. Last 80 lines:"
    tail -80 "$SEARCH_LOG" || true
    exit 1
  fi
  sleep 1
done
if [ "$SEARCH_OK" != "1" ]; then
  echo "ERROR: search server not reachable at ${SEARCH_NODE_IP}:18999. Last 80 lines:"
  tail -80 "$SEARCH_LOG" || true
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
    conda activate ${TRAIN_CONDA_ENV:-cxtgraph}
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
RAY_HEAD_LOG="$PROJECT_ROOT/logs/ray-head-${SLURM_JOB_ID:-idev}-${RUN_TAG}-ctxgraph.log"
probe "Ray head log: $RAY_HEAD_LOG"
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" bash -c '
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate ${TRAIN_CONDA_ENV:-cxtgraph}
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
  exec ray start --head --node-ip-address='"$TRAINER_HEAD_IP"' --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
' >"$RAY_HEAD_LOG" 2>&1 &
RAY_HEAD_PID=$!
sleep 20
probe "Ray head sleep done; launching $((NUM_NODES - 2)) trainer workers (skip search node)"

# ── Ray workers on NODELIST[1..N-1] ──
WORKER_PIDS=()
for i in $(seq 2 $((NUM_NODES - 1))); do  # skip NODELIST[0]=search, [1]=head
  WORKER_NODE=${NODELIST[$i]}
  RAY_WORKER_LOG="$PROJECT_ROOT/logs/ray-worker-${SLURM_JOB_ID:-idev}-${RUN_TAG}-ctxgraph-${i}.log"
  probe "Ray worker log ($WORKER_NODE): $RAY_WORKER_LOG"
  srun --overlap --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c '
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate ${TRAIN_CONDA_ENV:-cxtgraph}
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
    exec ray start --address='"${TRAINER_HEAD_IP}:6379"' --num-cpus=70 --num-gpus=1 --block
  ' >"$RAY_WORKER_LOG" 2>&1 &
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
echo "  Launching ContextGraph ${ADV_ESTIMATOR} + v5 ($MODEL_PATH, update=$MODEL_UPDATE_LABEL, $NUM_NODES nodes [1 search + $((NUM_NODES - 1)) trainer], $TOTAL_TRAINING_STEPS steps, BS=$TRAIN_BATCH_SIZE, rollout_n=$ROLLOUT_N, ppo_mini=$PPO_MINI_BATCH_SIZE, context=$CONTEXT_LENGTH, $DATASET_LABEL)"
echo "  Optimization: lr=$TRAIN_LR use_kl_loss=$USE_KL_LOSS clip=[$CLIP_RATIO_LOW,$CLIP_RATIO_HIGH]"
echo "  CG-specific: workflow=search_graph, process_reward=$PROCESS_REWARD_SPEC, lambda_compact=0.2, lambda_cost=0.02, consolidation K=5"
echo "  Graph protocol: $BC_CTXGRAPH_PROTOCOL structured_controller=$BC_STRUCTURED_GRAPH_CONTROLLER controller_formatting=$BC_CONTROLLER_OWNED_TOOL_FORMATTING action_policy=$BC_CONTROLLER_ACTION_POLICY"
echo "  Structured memory: enabled=$STRUCTURED_MEMORY_ENABLED context_budget=$STRUCTURED_MEMORY_CONTEXT_BUDGET max_facts=$STRUCTURED_MEMORY_MAX_FACTS gap_interval=$STRUCTURED_MEMORY_GAP_INTERVAL jitter=$STRUCTURED_MEMORY_GAP_JITTER retries=$STRUCTURED_MEMORY_CONTROLLER_RETRIES stop_on_ready=$STRUCTURED_MEMORY_STOP_ON_READY step_limit=$STRUCTURED_MEMORY_STEP_LIMIT"
if [ "$ADV_ESTIMATOR" = "graphrpo" ]; then
  echo "  GraphRPO credit: $GRAPH_RPO_CREDIT_BACKEND"
  if [ "$GRAPH_RPO_CREDIT_BACKEND" = "old_policy_counterfactual_qa" ]; then
    echo "  Counterfactual QA: samples/state=$GRAPH_RPO_COUNTERFACTUAL_SAMPLES max_tokens=$GRAPH_RPO_COUNTERFACTUAL_MAX_NEW_TOKENS temperature=$GRAPH_RPO_COUNTERFACTUAL_TEMPERATURE top_p=$GRAPH_RPO_COUNTERFACTUAL_TOP_P thinking=$GRAPH_RPO_COUNTERFACTUAL_ENABLE_THINKING"
  fi
fi
echo "  v5 add-ons: uniqueness_weight=0.10 (Improvement #1), auto_bind_branch_edges=True with min_overlap=0.05 (Improvement #3)"
echo "  vLLM gpu_memory_utilization=0.6 + FSDP CPU offload"
echo "  val_before_train=$VAL_BEFORE_TRAIN, save_freq=$SAVE_FREQ, val every $TEST_FREQ steps"
echo "=============================================================="
probe "launching trainer (model load + vLLM init typically ~3-5 min)"

set +e
srun --overlap --nodes=1 --ntasks=1 -w "$TRAINER_HEAD_NODE" --chdir="$PROJECT_ROOT" \
  --export=ALL,LOCAL_SEARCH_URL="$LOCAL_SEARCH_URL",OPENAI_API_KEY="$OPENAI_API_KEY",JUDGE_MODEL="$JUDGE_MODEL" \
  python -m scripts.train_graph \
  algorithm.adv_estimator="$ADV_ESTIMATOR" \
  algorithm.kl_ctrl.kl_coef="$ALGORITHM_KL_COEF" \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules="$LORA_TARGET_MODULES" \
  "${LONG_CONTEXT_ARGS[@]}" \
  actor_rollout_ref.rollout.prompt_length="$PROMPT_LENGTH" \
  actor_rollout_ref.rollout.response_length="$RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$CONTEXT_LENGTH" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_ROLLOUT_N" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
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
  actor_rollout_ref.actor.policy_loss.loss_mode="$POLICY_LOSS_MODE" \
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
  actor_rollout_ref.actor.grad_clip=0.5 \
  actor_rollout_ref.actor.kl_loss_coef="$ACTOR_KL_LOSS_COEF" \
  algorithm.foldgrpo_process_reward_mode=relative_extrema \
  data.train_files="$TRAIN_DATA_FILE" \
  data.val_files="$VAL_DATA_FILE" \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  "${DATA_SEED_ARGS[@]}" \
  data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
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
  +actor_rollout_ref.rollout.plugin.workflow=search_graph \
  +actor_rollout_ref.rollout.plugin.structured_graph_controller="$BC_STRUCTURED_GRAPH_CONTROLLER" \
  +actor_rollout_ref.rollout.plugin.controller_owned_tool_formatting="$BC_CONTROLLER_OWNED_TOOL_FORMATTING" \
  +actor_rollout_ref.rollout.plugin.controller_action_policy="$BC_CONTROLLER_ACTION_POLICY" \
  +actor_rollout_ref.rollout.plugin.structured_memory_enabled="$STRUCTURED_MEMORY_ENABLED" \
  +actor_rollout_ref.rollout.plugin.structured_memory_required="$STRUCTURED_MEMORY_REQUIRED" \
  +actor_rollout_ref.rollout.plugin.structured_memory_gap_interval="$STRUCTURED_MEMORY_GAP_INTERVAL" \
  +actor_rollout_ref.rollout.plugin.structured_memory_context_budget="$STRUCTURED_MEMORY_CONTEXT_BUDGET" \
  +actor_rollout_ref.rollout.plugin.structured_memory_max_context_facts="$STRUCTURED_MEMORY_MAX_CONTEXT_FACTS" \
  +actor_rollout_ref.rollout.plugin.structured_memory_max_facts_per_observation="$STRUCTURED_MEMORY_MAX_FACTS_PER_OBSERVATION" \
  +actor_rollout_ref.rollout.plugin.structured_memory_max_facts="$STRUCTURED_MEMORY_MAX_FACTS" \
  +actor_rollout_ref.rollout.plugin.structured_memory_extract_max_tokens="$STRUCTURED_MEMORY_EXTRACT_MAX_TOKENS" \
  +actor_rollout_ref.rollout.plugin.structured_memory_gap_max_tokens="$STRUCTURED_MEMORY_GAP_MAX_TOKENS" \
  +actor_rollout_ref.rollout.plugin.structured_memory_plan_max_tokens="$STRUCTURED_MEMORY_PLAN_MAX_TOKENS" \
  +actor_rollout_ref.rollout.plugin.structured_memory_relation_candidates="$STRUCTURED_MEMORY_RELATION_CANDIDATES" \
  +actor_rollout_ref.rollout.plugin.structured_memory_controller_retries="$STRUCTURED_MEMORY_CONTROLLER_RETRIES" \
  +actor_rollout_ref.rollout.plugin.structured_memory_gap_jitter="$STRUCTURED_MEMORY_GAP_JITTER" \
  +actor_rollout_ref.rollout.plugin.structured_memory_stop_on_ready="$STRUCTURED_MEMORY_STOP_ON_READY" \
  +actor_rollout_ref.rollout.plugin.structured_memory_step_limit="$STRUCTURED_MEMORY_STEP_LIMIT" \
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
  +actor_rollout_ref.rollout.plugin.process_reward="$PROCESS_REWARD_SPEC" \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.2 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.02 \
  +actor_rollout_ref.rollout.plugin.consolidation_interval="$CONSOLIDATION_INTERVAL" \
  +actor_rollout_ref.rollout.plugin.auto_prune_max_active="$AUTO_PRUNE_MAX_ACTIVE" \
  +actor_rollout_ref.rollout.plugin.graph_invalid_penalty=-0.3 \
  +actor_rollout_ref.rollout.plugin.max_traj=11 \
  +actor_rollout_ref.rollout.plugin.uniqueness_weight=0.10 \
  +actor_rollout_ref.rollout.plugin.auto_bind_branch_edges=True \
  +actor_rollout_ref.rollout.plugin.auto_bind_min_overlap=0.05 \
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
  "${ROLLOUT_DATA_ARGS[@]}" \
  "${VALIDATION_DATA_ARGS[@]}" \
  "${GRAPH_RPO_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "$@"
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
