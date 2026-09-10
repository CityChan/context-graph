#!/bin/bash
# QeRL/TRL colocated-vLLM training for FoldAgent or ContextGraph.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
QERL_ROOT=${QERL_ROOT:-/work/09281/chc_1996/vista/QeRL}
CONDA_BASE=${CONDA_BASE:-/work/09281/chc_1996/vista/miniconda3}
AGENT_KIND=${AGENT_KIND:-foldagent}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-$TOTAL_TRAINING_STEPS}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-0}
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-}
DATASET_NAME=${DATASET_NAME:-gsm8k}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-16}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}
TS=$(date +%Y%m%d_%H%M%S)
RUN_TAG=${RUN_TAG:-trl-${AGENT_KIND}-gsm8k-qwen25-1p5b-lora32-${TOTAL_TRAINING_STEPS}step}
RUN_NAME=${RUN_NAME:-${RUN_TAG}-${SLURM_JOB_ID:-local}-$TS}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${SCRATCH:?SCRATCH must be set}/context-graph-trl-ckpts/$RUN_NAME}
MASTER_PORT=${MASTER_PORT:-29539}
TRL_NUM_MACHINES=${TRL_NUM_MACHINES:-1}
TRL_NUM_PROCESSES=${TRL_NUM_PROCESSES:-$TRL_NUM_MACHINES}
TRL_MACHINE_RANK=${TRL_MACHINE_RANK:-${SLURM_PROCID:-0}}
TRL_MASTER_ADDR=${TRL_MASTER_ADDR:-localhost}

if [ "$TRL_NUM_MACHINES" -lt 1 ] || [ "$TRL_NUM_PROCESSES" -lt 1 ]; then
  echo "ERROR: TRL_NUM_MACHINES and TRL_NUM_PROCESSES must be positive" >&2
  exit 2
fi
if [ "$TRL_NUM_PROCESSES" -ne "$TRL_NUM_MACHINES" ]; then
  echo "ERROR: this launcher supports exactly one trainer process per machine" >&2
  exit 2
fi
if [ "$TRL_MACHINE_RANK" -lt 0 ] || [ "$TRL_MACHINE_RANK" -ge "$TRL_NUM_MACHINES" ]; then
  echo "ERROR: TRL_MACHINE_RANK=$TRL_MACHINE_RANK is outside [0,$((TRL_NUM_MACHINES - 1))]" >&2
  exit 2
fi

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
export PYTHONNOUSERSITE=1 CC=gcc CXX=g++ MASTER_ADDR="$TRL_MASTER_ADDR" MASTER_PORT
unset RANK LOCAL_RANK WORLD_SIZE
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN TORCHDYNAMO_DISABLE=1
export WANDB_PROJECT=${WANDB_PROJECT:-context-graph}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-trl-agent-framework}
export WANDB_TAGS=${WANDB_TAGS:-trl,agent,$AGENT_KIND,$DATASET_NAME,qwen2.5-1.5b,lora32,g16}
export WANDB_LOG_MODEL=false
export CUDA_VISIBLE_DEVICES=${TRL_AGENT_CUDA_DEVICE:-0}

TRAIN_DATA_ARGS=()
if [ -n "$TRAIN_DATA_PATH" ]; then
  if [ ! -s "$TRAIN_DATA_PATH" ]; then
    echo "ERROR: training parquet missing or empty: $TRAIN_DATA_PATH" >&2
    exit 1
  fi
  TRAIN_DATA_ARGS+=(--train-data-path "$TRAIN_DATA_PATH")
fi

python -c "import ctypes, omegaconf, tensordict, torch, trl, vllm, verl; ctypes.CDLL('libnvrtc.so.12'); assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print('runtime OK:', torch.__version__, trl.__version__, vllm.__version__)"
echo "TRL agent training: agent=$AGENT_KIND dataset=$DATASET_NAME model=$MODEL_PATH steps=$TOTAL_TRAINING_STEPS config=$AGENT_CONFIG machines=$TRL_NUM_MACHINES rank=$TRL_MACHINE_RANK"
echo "checkpoint=$CHECKPOINT_ROOT run=$RUN_NAME"

if [ "$TRL_NUM_MACHINES" -gt 1 ]; then
  ACCELERATE_CONFIG=$QERL_ROOT/recipes/accelerate_configs/ddp.yaml
  ACCELERATE_DISTRIBUTED_ARGS=(--num_machines "$TRL_NUM_MACHINES" --num_processes "$TRL_NUM_PROCESSES" --machine_rank "$TRL_MACHINE_RANK" --main_process_ip "$TRL_MASTER_ADDR")
else
  ACCELERATE_CONFIG=$QERL_ROOT/recipes/accelerate_configs/single_gpu.yaml
  ACCELERATE_DISTRIBUTED_ARGS=(--num_processes 1)
fi

accelerate launch --config_file "$ACCELERATE_CONFIG" "${ACCELERATE_DISTRIBUTED_ARGS[@]}" --main_process_port "$MASTER_PORT" -m trl_agent.train \
  --agent-kind "$AGENT_KIND" \
  --agent-config "$AGENT_CONFIG" \
  "${TRAIN_DATA_ARGS[@]}" \
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
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-completion-length "$MAX_COMPLETION_LENGTH" \
  --num-train-epochs 1 \
  --max-steps "$TOTAL_TRAINING_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-strategy steps \
  --save-total-limit 4 \
  --max-grad-norm 0.2 \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --lora-rank 32 \
  --lora-alpha 32 \
  --fast-inference True \
  --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm-tensor-parallel-size 1 \
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

if [ "$TRL_MACHINE_RANK" -ne 0 ]; then
  exit 0
fi

FINAL_CHECKPOINT=$CHECKPOINT_ROOT/checkpoint-$TOTAL_TRAINING_STEPS
test -s "$FINAL_CHECKPOINT/trainer_state.json"
python -c "import json, math; from pathlib import Path; p=Path('$FINAL_CHECKPOINT/trainer_state.json'); rows=[x for x in json.loads(p.read_text())['log_history'] if 'reward' in x]; assert rows, 'no reward metrics'; required={'loss','reward','reward/score','reward/task','reward_std','frac_reward_zero_std','reward/correctness','reward/correctness_reward','reward/soft_format_valid','reward/soft_format_reward','agent/vllm_generate_calls','training/old_policy_logps_recomputed','training/rollout_probs_diff_mean','actor/pg_loss','clip_ratio/region_mean'}; missing=sorted(required-set(rows[-1])); assert not missing, f'missing metrics: {missing}'; assert all(math.isfinite(float(rows[-1][k])) for k in required), 'non-finite final metrics'; assert rows[-1]['training/old_policy_logps_recomputed'] == 0.0, 'unexpected old-policy recomputation for aligned launch'; assert rows[-1]['clip_ratio/region_mean'] == 0.0, 'aligned one-iteration launch should not PPO-clip on vLLM numerical drift'; print('final metrics:', {k: rows[-1][k] for k in sorted(required)})"
if [ "$AGENT_KIND" = contextgraph ]; then
  python -c "import json; from pathlib import Path; p=Path('$FINAL_CHECKPOINT/trainer_state.json'); rows=[x for x in json.loads(p.read_text())['log_history'] if 'reward' in x]; required={'graphrpo/controller_attempts','graphrpo/controller_valid_edits','graphrpo/controller_errors','graphrpo/valid_edits','graphrpo/creditable_edits','graphrpo/credited_edits','graphrpo/scored_states','graphrpo/delta_abs_sum','graphrpo/counterfactual_probe_rollouts','graphrpo/counterfactual_tag_rate'}; missing=sorted(required-set().union(*(row.keys() for row in rows))); assert not missing, f'missing GraphRPO metrics: {missing}'; assert any(float(row.get('graphrpo/controller_valid_edits',0))>0 for row in rows), 'no valid controller edit reached the graph trace'; assert any(float(row.get('graphrpo/creditable_edits',0))>0 for row in rows), 'no graph edit reached counterfactual credit scoring'; assert any(float(row.get('graphrpo/scored_states',0))>0 for row in rows), 'no counterfactual graph state was scored'; assert any(float(row.get('graphrpo/counterfactual_probe_rollouts',0))>0 for row in rows), 'no counterfactual QA probe was generated'; assert any(float(row.get('graphrpo/counterfactual_tag_rate',0))>0 for row in rows), 'all counterfactual QA probes violated the answer contract'; print('GraphRPO plumbing audit: OK; inspect graphrpo/delta_abs_sum for utility signal')"
fi
echo "TRL agent training completed: $FINAL_CHECKPOINT"
