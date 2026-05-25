#!/bin/bash
# 2-node smoke validating the memory-polish knobs added to
# train_alfworld_ctxgraph_30b_8node_48h.sh.
#
# Scope:
#   - HF offline tokenizer load (proxy for the 704096 JSONDecodeError fix)
#   - vLLM accepts: gpu_memory_utilization=0.55, enforce_eager, max_num_seqs,
#                    free_cache_engine=False, engine_kwargs.vllm.enable_sleep_mode=False
#   - First rollout fits at prod-scale prompt=8192 / response=32768
#   - One save_freq=2 checkpoint dumps to /scratch
#   - sync_latest_checkpoint_to_hf.sh round-trips (opt-in)
#
# Designed to run INSIDE a 2-node idev allocation:
#     idev -p gh -N 2 -t 01:00:00 -A AST24021
#     bash scripts/smoke_alfworld_ctxgraph_30b_2node_gh.sh
#
# Override knobs via env vars at the top of the file as needed.

set -euo pipefail

# ── Allocation discovery (sbatch OR idev) ──
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
  ALLOC_JOB_ID="$SLURM_JOB_ID"
  NODELIST_SPEC="$SLURM_JOB_NODELIST"
  SRUN_PREFIX=(srun --overlap)
elif [ -n "${IDEV_JOBID:-}" ]; then
  ALLOC_JOB_ID="$IDEV_JOBID"
  NODELIST_SPEC=$(squeue -j "$ALLOC_JOB_ID" -h -o "%N" | head -n 1)
  SRUN_PREFIX=(srun --jobid="$ALLOC_JOB_ID" --overlap)
else
  echo "Run inside a Slurm allocation (idev or sbatch). Found neither SLURM_JOB_ID nor IDEV_JOBID."
  exit 1
fi

mapfile -t NODELIST < <(scontrol show hostnames "$NODELIST_SPEC")
NUM_NODES=${#NODELIST[@]}
if [ "$NUM_NODES" -ne 2 ]; then
  echo "Expected 2 nodes, got $NUM_NODES (${NODELIST[*]})"
  exit 1
fi

# ── Memory polish env vars (same as patched 48h prod script) ──
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export RAY_memory_usage_threshold=0.99
export RAY_memory_monitor_refresh_ms=0
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$$
export FLASHINFER_WORKSPACE_BASE=/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_FILE_LOCKING=1

# ── Conda + CUDA + toolchain ──
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph

export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:${LIBRARY_PATH:-}
export CPATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/include:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/include:${CPATH:-}

export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++
export TORCHDYNAMO_DISABLE=1
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_LEVEL=NVL

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"

export HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
# Vista quirk: weights live at $HF_HOME/models--XXX, not $HF_HOME/hub/.
# Pin HF_HUB_CACHE so new `hf`/transformers tooling reads the existing cache.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

export WANDB_API_KEY=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

# WANDB optional — if no key, fall back to console-only
if [ -n "${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER='["console","wandb"]'
else
  TRAINER_LOGGER='["console"]'
fi

NODE0=${NODELIST[0]}
NODE0_IP=$(getent hosts "$NODE0" | awk '{print $1}')
TS=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ctxgraph_alfworld_hard_30b_2n_smoke_${TS}}

# ── Smoke-scale config (memory regime kept at prod) ──
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-3}
SAVE_FREQ=${SAVE_FREQ:-2}        # triggers one save at step 2
TEST_FREQ=${TEST_FREQ:--1}        # skip val
DATA_N_TRAIN=${DATA_N_TRAIN:-64}
DATA_N_VAL=${DATA_N_VAL:-16}

# Memory stressors KEPT at prod scale
ROLLOUT_PROMPT_LENGTH=${ROLLOUT_PROMPT_LENGTH:-8192}
ROLLOUT_RESPONSE_LENGTH=${ROLLOUT_RESPONSE_LENGTH:-32768}
ROLLOUT_LOG_PROB_MAX_LEN=${ROLLOUT_LOG_PROB_MAX_LEN:-40960}

# Scaled DOWN for 2-node speed (rollout.n=2 vs prod 8, batch=4 vs 32)
ROLLOUT_N=${ROLLOUT_N:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}    # batch×n=8 total → 4/GPU

# Session/turn scaled down to bound wall time
PLUGIN_MAX_TURN=${PLUGIN_MAX_TURN:-20}
PLUGIN_MAX_SESSION=${PLUGIN_MAX_SESSION:-4}
PLUGIN_MAX_TRAJ=${PLUGIN_MAX_TRAJ:-4}

# Memory knobs being validated (same defaults as patched prod)
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}

# Checkpoint + HF sync
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/09281/chc_1996/vista/checkpoints/context-graph/${EXPERIMENT_NAME}}
HF_SYNC_CHECKPOINTS=${HF_SYNC_CHECKPOINTS:-0}
HF_NAMESPACE=${HF_NAMESPACE:-}
HF_REPO_ID=${HF_REPO_ID:-${HF_NAMESPACE}/${EXPERIMENT_NAME}}
HF_REPO_PRIVATE=${HF_REPO_PRIVATE:-true}

mkdir -p "$ALFWORLD_DATA"
mkdir -p "$(dirname "$CHECKPOINT_ROOT")"

echo "=============================================================="
echo "2-node memory-knob smoke for 30B ALFWorld @hard ctxgraph"
echo "Allocation: $ALLOC_JOB_ID   Nodes: ${NODELIST[*]}"
echo "Experiment: $EXPERIMENT_NAME"
echo "Steps=$TOTAL_TRAINING_STEPS save_freq=$SAVE_FREQ test_freq=$TEST_FREQ"
echo "Memory regime: prompt=$ROLLOUT_PROMPT_LENGTH resp=$ROLLOUT_RESPONSE_LENGTH gpu_util=$ROLLOUT_GPU_MEMORY_UTILIZATION eager=$ROLLOUT_ENFORCE_EAGER max_seqs=$ROLLOUT_MAX_NUM_SEQS"
echo "Batch: train=$TRAIN_BATCH_SIZE n=$ROLLOUT_N mini=$PPO_MINI_BATCH_SIZE"
echo "Plugin: max_turn=$PLUGIN_MAX_TURN max_session=$PLUGIN_MAX_SESSION max_traj=$PLUGIN_MAX_TRAJ"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "HF sync: $HF_SYNC_CHECKPOINTS (repo=$HF_REPO_ID)"
echo "Started: $(date)"
echo "=============================================================="

# ── Preflight 1: tokenizer offline load (catches 704096 root cause early) ──
echo "--- Preflight: HF offline tokenizer load ---"
python -c "
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-30B-A3B-Thinking-2507')
print('tokenizer vocab_size =', tok.vocab_size)
print('OK: offline tokenizer load succeeded')
"

# ── Preflight 2: env sanity ──
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import verl; print('verl OK')"

# ── ALFWorld data ──
echo "--- Generating ALFWorld parquet (hard) ---"
python scripts/make_alfworld_data.py --hard --n_train "$DATA_N_TRAIN" --n_val "$DATA_N_VAL"
python -c "
import pandas as pd
df = pd.read_parquet('data/alfworld_graph_train.parquet')
assert df['ability'].iloc[0] == 'ALFWorld@hard', f'expected hard, got {df[\"ability\"].iloc[0]}'
print(f'OK: graph_train.parquet ability={df[\"ability\"].iloc[0]} rows={len(df)}')
"

# ── Stale Ray cleanup ──
echo "--- Stopping any stale Ray processes ---"
for node in "${NODELIST[@]}"; do
  "${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$node" bash -c '
    source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
    conda activate cxtgraph
    ray stop -f >/dev/null 2>&1 || true
  ' || true
done
sleep 3

# ── Ray head on node0 ──
echo "--- Starting Ray head on $NODE0 ($NODE0_IP) ---"
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$NODE0" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
  export HF_HOME=$HF_HOME
  export HF_HUB_CACHE=$HF_HUB_CACHE
  export ALFWORLD_DATA=$ALFWORLD_DATA
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_FILE_LOCKING=1
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ray start --head --node-ip-address=$NODE0_IP --port=6379 \
    --num-cpus=70 --num-gpus=1 --dashboard-host=0.0.0.0 --block
" &
RAY_HEAD_PID=$!
sleep 20

# ── Ray worker on node1 ──
WORKER_NODE=${NODELIST[1]}
echo "--- Starting Ray worker on $WORKER_NODE ---"
"${SRUN_PREFIX[@]}" --nodes=1 --ntasks=1 -w "$WORKER_NODE" bash -c "
  source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
  conda activate cxtgraph
  export PATH=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/bin:\${PATH}
  export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8/lib64:\${LD_LIBRARY_PATH}
  export HF_HOME=$HF_HOME
  export HF_HUB_CACHE=$HF_HUB_CACHE
  export ALFWORLD_DATA=$ALFWORLD_DATA
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_DISABLE_FILE_LOCKING=1
  export FLASHINFER_WORKSPACE_BASE=/tmp
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ray start --address=${NODE0_IP}:6379 --num-cpus=70 --num-gpus=1 --block
" &
WORKER_PID=$!
sleep 15

cleanup() {
  kill "$RAY_HEAD_PID" 2>/dev/null || true
  kill "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT

export RAY_ADDRESS=${NODE0_IP}:6379
echo "--- Ray cluster status ---"
ray status || echo "WARN: ray status check failed"

# ── Training (scaled-down config, prod memory regime) ──
echo "=============================================================="
echo "Launching $TOTAL_TRAINING_STEPS-step smoke training"
echo "=============================================================="

set +e
python -m scripts.train_graph \
  algorithm.adv_estimator=foldgrpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  actor_rollout_ref.rollout.agent.default_agent_loop=context_graph_isolated_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=Qwen/Qwen3-30B-A3B-Thinking-2507 \
  actor_rollout_ref.rollout.prompt_length=${ROLLOUT_PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${ROLLOUT_RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_LEN} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.rollout.quantization=fp8 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
  actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
  actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS} \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_sleep_mode=False \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  data.train_files=data/alfworld_graph_train.parquet \
  data.val_files=data/alfworld_graph_test.parquet \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${ROLLOUT_PROMPT_LENGTH} \
  data.max_response_length=${ROLLOUT_RESPONSE_LENGTH} \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_LEN} \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_LEN} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  +actor_rollout_ref.rollout.plugin.workflow=alfworld_graph \
  +actor_rollout_ref.rollout.plugin.max_turn=${PLUGIN_MAX_TURN} \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.max_session=${PLUGIN_MAX_SESSION} \
  +actor_rollout_ref.rollout.plugin.val_max_session=${PLUGIN_MAX_SESSION} \
  +actor_rollout_ref.rollout.plugin.session_timeout=1800 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=8192 \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope,graph]' \
  +actor_rollout_ref.rollout.plugin.lambda_compact=0.1 \
  +actor_rollout_ref.rollout.plugin.lambda_cost=0.005 \
  +actor_rollout_ref.rollout.plugin.max_traj=${PLUGIN_MAX_TRAJ} \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=False \
  +actor_rollout_ref.rollout.plugin.val_max_turn=${PLUGIN_MAX_TURN} \
  +actor_rollout_ref.rollout.plugin.val_response_length=${ROLLOUT_RESPONSE_LENGTH} \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=${NUM_NODES} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.default_local_dir="$CHECKPOINT_ROOT" \
  trainer.project_name=context-graph \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger="$TRAINER_LOGGER"
RC=$?
set -e

echo "=============================================================="
echo "Training exit code: $RC"
echo "Local checkpoint dir contents:"
ls -lah "$CHECKPOINT_ROOT" 2>/dev/null || echo "(checkpoint root missing)"
echo "=============================================================="

# ── HF sync (opt-in) ──
if [ "$HF_SYNC_CHECKPOINTS" = "1" ] && [ "$RC" -eq 0 ]; then
  if [ -z "$HF_NAMESPACE" ]; then
    echo "HF_SYNC_CHECKPOINTS=1 but HF_NAMESPACE empty; skipping sync."
  else
    echo "--- Verifying HF auth ---"
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf auth whoami
    echo "--- Syncing latest checkpoint to HF: $HF_REPO_ID ---"
    bash scripts/sync_latest_checkpoint_to_hf.sh "$CHECKPOINT_ROOT" "$HF_REPO_ID" "$EXPERIMENT_NAME" "$HF_REPO_PRIVATE"
  fi
else
  echo "HF sync skipped (HF_SYNC_CHECKPOINTS=$HF_SYNC_CHECKPOINTS, RC=$RC)."
  echo "To enable: HF_SYNC_CHECKPOINTS=1 HF_NAMESPACE=<your-hf-user> bash $0"
fi

echo "Finished: $(date)"
exit $RC
