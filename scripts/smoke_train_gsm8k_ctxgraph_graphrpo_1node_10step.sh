#!/bin/bash

# True ContextGraph smoke on a simple, deterministic GSM8K task.
# This exercises MathEnv -> isolated ContextGraph agent -> structured graph
# controller -> GraphRPO -> actor optimizer, without search or an LLM judge.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
GSM8K_DATA_DIR=${GSM8K_DATA_DIR:-${SCRATCH:?SCRATCH must be set}/context-graph-data/gsm8k-contextgraph}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8}
CUDA_TARGET_LIB=$CUDA_HOME/targets/sbsa-linux/lib
CUDA_LIB=$CUDA_HOME/lib64
CUDA_INCLUDE=$CUDA_HOME/include
NVPL_INCLUDE=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
TS=$(date +%Y%m%d_%H%M%S)
RUN_TAG=${RUN_TAG:-gsm8k-qwen25-1p5b-ctxgraph-graphrpo-10step}
CACHE_TAG=${SLURM_JOB_ID:-local}-$TS-$$
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/contextgraph-gsm8k-cg-vllm-$CACHE_TAG}
TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/contextgraph-gsm8k-cg-inductor-$CACHE_TAG}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/contextgraph-gsm8k-cg-triton-$CACHE_TAG}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/contextgraph-gsm8k-cg-xdg-$CACHE_TAG}
CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/tmp/contextgraph-gsm8k-cg-cuda-$CACHE_TAG}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH/context-graph-ckpts/$RUN_TAG-$TS}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$SCRATCH/context-graph-rollouts/$RUN_TAG-$TS}
RUN_LOG=${RUN_LOG:-$PROJECT_ROOT/logs/$RUN_TAG-$TS.log}

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this inside an active Vista idev allocation."
  exit 2
fi

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"
mkdir -p logs "$GSM8K_DATA_DIR" "$CHECKPOINT_ROOT" "$ROLLOUT_DATA_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

if [ ! -e "$CUDA_TARGET_LIB/libnvrtc.so.12" ] && [ ! -e "$CUDA_LIB/libnvrtc.so.12" ]; then
  echo "ERROR: libnvrtc.so.12 is missing under $CUDA_HOME."
  exit 2
fi

export CUDA_HOME
export CUDACXX="$CUDA_HOME/bin/nvcc"
export CC=${GSM8K_CC:-gcc}
export CXX=${GSM8K_CXX:-g++}
export CUDAHOSTCXX=${GSM8K_CUDAHOSTCXX:-g++}
export PATH="${CONDA_PREFIX}/bin:$CUDA_HOME/bin:$PATH"
hash -r
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:$CUDA_TARGET_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_TARGET_LIB:$CUDA_LIB:${LIBRARY_PATH:-}"
export CPATH="$NVPL_INCLUDE:$CUDA_INCLUDE:${CPATH:-}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}"

VISIBLE_GPUS=$(nvidia-smi -L | wc -l)
NUM_GPUS=${NUM_GPUS:-$VISIBLE_GPUS}
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
  echo "ERROR: NUM_GPUS must be a positive integer, got: $NUM_GPUS"
  exit 2
fi
if [ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]; then
  echo "ERROR: requested $NUM_GPUS GPUs but only $VISIBLE_GPUS are visible on $(hostname)."
  exit 2
fi

export HF_HOME HF_HUB_CACHE VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR XDG_CACHE_HOME CUDA_CACHE_PATH
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_DISABLE_FILE_LOCKING=1
export TOKENIZERS_PARALLELISM=false TORCHDYNAMO_DISABLE=1 HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_P2P_LEVEL=NVL
export RAY_memory_usage_threshold=0.99 RAY_memory_monitor_refresh_ms=0
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE RAY_ADDRESS OPENAI_API_KEY OPENAI_URL LOCAL_SEARCH_URL
ray stop --force >/dev/null 2>&1 || true

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for command_name in nvidia-smi gcc g++ nvcc; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: required command is unavailable: $command_name"
    exit 2
  fi
done

python -c "import ctypes; ctypes.CDLL('libnvrtc.so.12'); print('libnvrtc.so.12: OK')"
python -c "import torch; count=torch.cuda.device_count(); print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available(), 'devices:', count); assert torch.cuda.is_available() and count >= int('$NUM_GPUS')"
python -c "import vllm, verl; print('vllm:', vllm.__version__, 'verl: OK')"

if [ ! -s "$GSM8K_DATA_DIR/train.parquet" ] || [ ! -s "$GSM8K_DATA_DIR/test.parquet" ]; then
  python scripts/prepare_gsm8k_grpo_data.py --output-dir "$GSM8K_DATA_DIR"
fi
if ! python -c "import pandas as pd; extra=pd.read_parquet('$GSM8K_DATA_DIR/train.parquet').iloc[0]['extra_info']; assert extra['workflow']=='math_graph' and extra['reward_mode']=='gsm8k_exact' and extra['query'] and extra['answer']"; then
  echo "Refreshing stale GSM8K data with ContextGraph metadata."
  python scripts/prepare_gsm8k_grpo_data.py --output-dir "$GSM8K_DATA_DIR"
fi
python -c "import pandas as pd; extra=pd.read_parquet('$GSM8K_DATA_DIR/train.parquet').iloc[0]['extra_info']; assert extra['workflow']=='math_graph' and extra['reward_mode']=='gsm8k_exact' and extra['query'] and extra['answer']; print('ContextGraph GSM8K schema: OK')"

echo "GSM8K ContextGraph GraphRPO smoke: model=$MODEL_PATH gpus=$NUM_GPUS steps=$TOTAL_TRAINING_STEPS"
echo "log=$RUN_LOG"
echo "checkpoints=$CHECKPOINT_ROOT"
echo "rollouts=$ROLLOUT_DATA_DIR"
echo "protocol=MathEnv + math_graph + context_graph_isolated_agent + structured GraphRPO"

set +e
python -m scripts.train_graph \
  algorithm.adv_estimator=graphrpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.graphrpo_alpha=0.1 \
  algorithm.graphrpo_beta=1.0 \
  algorithm.graphrpo_epsilon=1e-6 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.policy_loss.loss_mode=graphrpo \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.prompt_length=1024 \
  actor_rollout_ref.rollout.response_length=2048 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  +actor_rollout_ref.rollout.plugin.workflow=math_graph \
  +actor_rollout_ref.rollout.plugin.structured_graph_controller=True \
  +actor_rollout_ref.rollout.plugin.controller_action_policy=structural \
  +actor_rollout_ref.rollout.plugin.graph_rpo_credit_backend=old_policy_answer_likelihood \
  +actor_rollout_ref.rollout.plugin.graph_rpo_delta_max=0.25 \
  +actor_rollout_ref.rollout.plugin.graph_rpo_reference_max_prompt_length=1024 \
  +actor_rollout_ref.rollout.plugin.graph_rpo_reference_max_answer_length=64 \
  +actor_rollout_ref.rollout.plugin.graph_rpo_reference_enable_thinking=False \
  +actor_rollout_ref.rollout.plugin.graph_rpo_scope_process_reward=False \
  +actor_rollout_ref.rollout.plugin.process_reward='[graphrpo]' \
  +actor_rollout_ref.rollout.plugin.must_branch=True \
  +actor_rollout_ref.rollout.plugin.max_session=2 \
  +actor_rollout_ref.rollout.plugin.val_max_session=2 \
  +actor_rollout_ref.rollout.plugin.max_turn=8 \
  +actor_rollout_ref.rollout.plugin.val_max_turn=8 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=256 \
  +actor_rollout_ref.rollout.plugin.branch_len=1024 \
  +actor_rollout_ref.rollout.plugin.final_answer_reserve=192 \
  +actor_rollout_ref.rollout.plugin.final_answer_safety_margin=64 \
  +actor_rollout_ref.rollout.plugin.initial_consolidation_turn=1 \
  +actor_rollout_ref.rollout.plugin.consolidation_interval=1 \
  +actor_rollout_ref.rollout.plugin.graph_controller_min_completion_tokens=128 \
  +actor_rollout_ref.rollout.plugin.max_traj=3 \
  +actor_rollout_ref.rollout.plugin.must_finish=True \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_response_length=2048 \
  data.train_files="$GSM8K_DATA_DIR/train.parquet" \
  data.val_files="$GSM8K_DATA_DIR/test.parquet" \
  data.train_max_samples=128 \
  data.val_max_samples=32 \
  data.train_batch_size=4 \
  data.max_prompt_length=1024 \
  data.max_response_length=2048 \
  data.return_raw_chat=True \
  reward_manager.name=naive \
  trainer.val_before_train=True \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq=5 \
  trainer.save_freq="$TOTAL_TRAINING_STEPS" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.project_name=context-graph \
  trainer.experiment_name="$RUN_TAG-$TS" \
  trainer.logger=console \
  trainer.default_local_dir="$CHECKPOINT_ROOT" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.resume_mode=disable 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
  echo "GSM8K ContextGraph GraphRPO smoke failed with exit code $RC; log=$RUN_LOG"
  exit "$RC"
fi

if [ ! -d "$CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS/actor" ]; then
  echo "ERROR: actor checkpoint is missing: $CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS/actor"
  exit 1
fi
grep -q 'actor/pg_loss:' "$RUN_LOG" || { echo "ERROR: actor policy loss was not logged"; exit 1; }
grep -q 'actor/grad_norm:' "$RUN_LOG" || { echo "ERROR: actor optimizer step was not logged"; exit 1; }
grep -Eq 'reward/graph_explicit_ops[^0-9]*([1-9]|0\.[0-9]*[1-9])' "$RUN_LOG" || { echo "ERROR: no explicit ContextGraph operation was observed"; exit 1; }
grep -Eq 'graphrpo/old_policy_scored_states:[1-9]' "$RUN_LOG" || { echo "ERROR: GraphRPO did not score graph states"; exit 1; }

echo "Key training evidence:"
grep -E 'training/global_step:|reward/task_reward|reward/graph_explicit_ops|old_policy_scored_states|actor/pg_loss|actor/grad_norm|actor/kl_loss' "$RUN_LOG" | tail -30
echo "GSM8K ContextGraph GraphRPO smoke completed; checkpoint=$CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS log=$RUN_LOG"
