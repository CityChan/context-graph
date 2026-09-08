#!/bin/bash

# Minimal rule-reward GRPO benchmark for one Vista GH200 compute node.
# Run this inside an active idev allocation; no search server or API key is used.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
GSM8K_DATA_DIR=${GSM8K_DATA_DIR:-${SCRATCH:?SCRATCH must be set}/context-graph-data/gsm8k}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
CUDA_HOME=${CUDA_HOME:-/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8}
CUDA_TARGET_LIB=$CUDA_HOME/targets/sbsa-linux/lib
CUDA_LIB=$CUDA_HOME/lib64
CUDA_INCLUDE=$CUDA_HOME/include
NVPL_INCLUDE=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PROMPT_LENGTH=${PROMPT_LENGTH:-512}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-512}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-512}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-5}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
TRAIN_LR=${TRAIN_LR:-1e-6}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
OPTIMIZER=${OPTIMIZER:-AdamW}
OPTIMIZER_IMPL=${OPTIMIZER_IMPL:-torch.optim}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-constant}
CLIP_GRAD=${CLIP_GRAD:-1.0}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
DATA_PROMPT_STYLE=${DATA_PROMPT_STYLE:-hash}
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-null}
CUSTOM_REWARD_NAME=${CUSTOM_REWARD_NAME:-compute_score}
TRAINER_LOGGER=${TRAINER_LOGGER:-console}
TS=$(date +%Y%m%d_%H%M%S)
RUN_TAG=${RUN_TAG:-gsm8k-qwen25-1p5b-grpo-10step-fixed}
CACHE_TAG=${SLURM_JOB_ID:-local}-$TS-$$
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/contextgraph-gsm8k-vllm-$CACHE_TAG}
TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/contextgraph-gsm8k-inductor-$CACHE_TAG}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/contextgraph-gsm8k-triton-$CACHE_TAG}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/contextgraph-gsm8k-xdg-$CACHE_TAG}
CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-/tmp/contextgraph-gsm8k-cuda-$CACHE_TAG}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$SCRATCH/context-graph-ckpts/$RUN_TAG-$TS}
RUN_LOG=${RUN_LOG:-$PROJECT_ROOT/logs/$RUN_TAG-$TS.log}

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  echo "ERROR: run this inside an active Vista idev allocation."
  exit 2
fi

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
cd "$PROJECT_ROOT"
mkdir -p logs "$GSM8K_DATA_DIR" "$CHECKPOINT_ROOT" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

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
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NUM_GPUS must be a positive integer, got: $NUM_GPUS"
  exit 2
fi
if [ "$NUM_GPUS" -lt 1 ]; then
  echo "ERROR: no GPU is visible on $(hostname)."
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
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE RAY_ADDRESS
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
  python scripts/prepare_gsm8k_grpo_data.py --output-dir "$GSM8K_DATA_DIR" --prompt-style "$DATA_PROMPT_STYLE"
fi
python -c "import pandas as pd; p='$DATA_PROMPT_STYLE'; rows=pd.read_parquet('$GSM8K_DATA_DIR/train.parquet'); actual=rows.iloc[0]['extra_info'].get('prompt_style', 'hash'); assert actual == p, f'prompt style mismatch: expected {p}, got {actual}'"

echo "GSM8K GRPO smoke: model=$MODEL_PATH gpus=$NUM_GPUS steps=$TOTAL_TRAINING_STEPS"
echo "log=$RUN_LOG"
echo "checkpoints=$CHECKPOINT_ROOT"
echo "node-local caches: vllm=$VLLM_CACHE_ROOT inductor=$TORCHINDUCTOR_CACHE_DIR triton=$TRITON_CACHE_DIR"
echo "runtime: CC=$CC CXX=$CXX CUDA_HOME=$CUDA_HOME VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD"
echo "batching=train_batch_size=$TRAIN_BATCH_SIZE rollout_n=$ROLLOUT_N ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
echo "model_update=lora_rank=$LORA_RANK lora_alpha=$LORA_ALPHA lr=$TRAIN_LR optimizer=$OPTIMIZER scheduler=$LR_SCHEDULER_TYPE"
echo "reward=prompt_style=$DATA_PROMPT_STYLE custom_path=$CUSTOM_REWARD_PATH"

set +e
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank="$LORA_RANK" \
  actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr="$TRAIN_LR" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.optim.lr_scheduler_type="$LR_SCHEDULER_TYPE" \
  actor_rollout_ref.actor.optim.optimizer="$OPTIMIZER" \
  actor_rollout_ref.actor.optim.optimizer_impl="$OPTIMIZER_IMPL" \
  actor_rollout_ref.actor.optim.weight_decay="$WEIGHT_DECAY" \
  actor_rollout_ref.actor.optim.betas="[$ADAM_BETA1,$ADAM_BETA2]" \
  actor_rollout_ref.actor.optim.clip_grad="$CLIP_GRAD" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW" \
  actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
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
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_num_seqs=64 \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  data.train_files="$GSM8K_DATA_DIR/train.parquet" \
  data.val_files="$GSM8K_DATA_DIR/test.parquet" \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$PROMPT_LENGTH" \
  data.max_response_length="$RESPONSE_LENGTH" \
  data.return_raw_chat=True \
  reward_manager.name=naive \
  custom_reward_function.path="$CUSTOM_REWARD_PATH" \
  custom_reward_function.name="$CUSTOM_REWARD_NAME" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.project_name=context-graph \
  trainer.experiment_name="$RUN_TAG-$TS" \
  trainer.logger="$TRAINER_LOGGER" \
  trainer.default_local_dir="$CHECKPOINT_ROOT" \
  trainer.resume_mode=disable 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e

if [ "$RC" -ne 0 ]; then
  echo "GSM8K GRPO smoke failed with exit code $RC; log=$RUN_LOG"
  exit "$RC"
fi

echo "GSM8K GRPO smoke completed; log=$RUN_LOG"
