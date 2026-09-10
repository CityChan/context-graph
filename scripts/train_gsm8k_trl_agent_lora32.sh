#!/bin/bash
# QeRL/TRL colocated-vLLM training for FoldAgent or ContextGraph on one GH200.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
QERL_ROOT=${QERL_ROOT:-/work/09281/chc_1996/vista/QeRL}
CONDA_BASE=${CONDA_BASE:-/work/09281/chc_1996/vista/miniconda3}
AGENT_KIND=${AGENT_KIND:-foldagent}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-$TOTAL_TRAINING_STEPS}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-0}
TS=$(date +%Y%m%d_%H%M%S)
RUN_TAG=${RUN_TAG:-trl-${AGENT_KIND}-gsm8k-qwen25-1p5b-lora32-${TOTAL_TRAINING_STEPS}step}
RUN_NAME=${RUN_NAME:-${RUN_TAG}-${SLURM_JOB_ID:-local}-$TS}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SCRATCH:?SCRATCH must be set}/context-graph-trl-ckpts/$RUN_NAME}
MASTER_PORT=${MASTER_PORT:-29539}

case "$AGENT_KIND" in
  foldagent) AGENT_CONFIG=${AGENT_CONFIG:-$PROJECT_ROOT/recipes/trl_agent/foldagent_gsm8k.yaml} ;;
  contextgraph) AGENT_CONFIG=${AGENT_CONFIG:-$PROJECT_ROOT/recipes/trl_agent/contextgraph_gsm8k.yaml} ;;
  *) echo "ERROR: AGENT_KIND must be foldagent or contextgraph" >&2; exit 2 ;;
esac

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate graphtrl
cd "$PROJECT_ROOT"
mkdir -p logs "$CHECKPOINT_ROOT"

export PATH="/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LIBRARY_PATH:-}"
export CPATH="/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include:${CPATH:-}"
export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
export TRANSFORMERS_CACHE="$HF_HOME"
export PYTHONPATH="$PROJECT_ROOT:$QERL_ROOT:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 CC=gcc CXX=g++ RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=localhost MASTER_PORT
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT=${WANDB_PROJECT:-context-graph}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-trl-agent-framework}
export WANDB_TAGS=${WANDB_TAGS:-trl,agent,$AGENT_KIND,gsm8k,qwen2.5-1.5b,lora32,g16}
export WANDB_LOG_MODEL=false
export CUDA_VISIBLE_DEVICES=${TRL_AGENT_CUDA_DEVICE:-0}

python -c "import ctypes, omegaconf, tensordict, torch, trl, vllm, verl; ctypes.CDLL('libnvrtc.so.12'); assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print('runtime OK:', torch.__version__, trl.__version__, vllm.__version__)"
echo "TRL agent training: agent=$AGENT_KIND model=$MODEL_PATH steps=$TOTAL_TRAINING_STEPS config=$AGENT_CONFIG"
echo "checkpoint=$CHECKPOINT_ROOT run=$RUN_NAME"

accelerate launch --config_file "$QERL_ROOT/recipes/accelerate_configs/single_gpu.yaml" --num_processes=1 --main_process_port "$MASTER_PORT" -m trl_agent.train \
  --agent-kind "$AGENT_KIND" \
  --agent-config "$AGENT_CONFIG" \
  --train-max-samples "$TRAIN_MAX_SAMPLES" \
  --model-name "$MODEL_PATH" \
  --output-dir "$CHECKPOINT_ROOT" \
  --use-vllm True \
  --learning-rate 1e-5 \
  --adam-beta1 0.9 \
  --adam-beta2 0.99 \
  --weight-decay 0.1 \
  --warmup-ratio 0.1 \
  --lr-scheduler-type cosine \
  --optim adamw_8bit \
  --logging-steps 1 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --num-generations 16 \
  --max-prompt-length 1024 \
  --max-completion-length 2048 \
  --num-train-epochs 1 \
  --max-steps "$TOTAL_TRAINING_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-strategy steps \
  --save-total-limit 4 \
  --max-grad-norm 0.2 \
  --max-seq-length 4096 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --fast-inference True \
  --vllm-gpu-memory-utilization 0.45 \
  --random-state 2025 \
  --loss-type grpo \
  --beta 0.0 \
  --epsilon-high 0.28 \
  --num-iterations 1 \
  --mask-truncated-completions True \
  --run-name "$RUN_NAME" \
  --ln False \
  --disable-noise True \
  --vllm-enable-sleep-mode False

FINAL_CHECKPOINT=$CHECKPOINT_ROOT/checkpoint-$TOTAL_TRAINING_STEPS
test -s "$FINAL_CHECKPOINT/trainer_state.json"
python -c "import json, math; from pathlib import Path; p=Path('$FINAL_CHECKPOINT/trainer_state.json'); rows=[x for x in json.loads(p.read_text())['log_history'] if 'reward' in x]; assert rows, 'no reward metrics'; required={'loss','reward','reward/score','reward_std','frac_reward_zero_std','reward/correctness','reward/correctness_reward','reward/soft_format_valid','reward/soft_format_reward','agent/vllm_generate_calls','training/old_policy_logps_recomputed','training/rollout_probs_diff_mean','actor/pg_loss','clip_ratio/region_mean'}; missing=sorted(required-set(rows[-1])); assert not missing, f'missing metrics: {missing}'; assert all(math.isfinite(float(rows[-1][k])) for k in required), 'non-finite final metrics'; assert rows[-1]['training/old_policy_logps_recomputed'] == 0.0, 'unexpected old-policy recomputation for aligned launch'; assert rows[-1]['clip_ratio/region_mean'] == 0.0, 'aligned one-iteration launch should not PPO-clip on vLLM numerical drift'; print('final metrics:', {k: rows[-1][k] for k in sorted(required)})"
echo "TRL agent training completed: $FINAL_CHECKPOINT"
