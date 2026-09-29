#!/bin/bash
# Shared batch entry point; submit via submit_train_bcp_qwen35_9b_50step.sh.
set -euo pipefail
METHOD=${1:?Expected contextgraph or foldagent}
case "$METHOD" in contextgraph|foldagent) ;; *) echo "Invalid method: $METHOD" >&2; exit 2 ;; esac
export BCP_TRAIN_PROFILE=${BCP_TRAIN_PROFILE:-default}
case "$BCP_TRAIN_PROFILE" in
  default) ;;
  foldagent_32k_paper_batch)
    [ "$METHOD" = foldagent ] && [ "${SMOKE_TEST:-0}" != 1 ] || { echo "FoldAgent paper-batch profile requires foldagent normal training" >&2; exit 2; }
    ;;
  contextgraph_32k_paper_batch|contextgraph_64k_paper_batch)
    [ "$METHOD" = contextgraph ] && [ "${SMOKE_TEST:-0}" != 1 ] || { echo "Paper-batch profile requires contextgraph normal training" >&2; exit 2; }
    ;;
  *) echo "Unknown BCP_TRAIN_PROFILE: $BCP_TRAIN_PROFILE" >&2; exit 2 ;;
esac
: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${SCRATCH:?Vista SCRATCH must be set}"
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
ALLOCATED_NODE_COUNT=${#NODES[@]}
UNUSED_NODES=()
unset BC_ACTIVE_NODE_COUNT
export EXPECTED_NUM_NODES=5
PLANNED_STEPS=50
RUN_MODE=train
if [[ "$BCP_TRAIN_PROFILE" = *_32k_paper_batch ]]; then RUN_MODE=train32k; fi
if [ "$BCP_TRAIN_PROFILE" = contextgraph_64k_paper_batch ]; then RUN_MODE=train64k; fi
if [ "${SMOKE_TEST:-0}" = 1 ]; then
  export EXPECTED_NUM_NODES=4
  PLANNED_STEPS=1
  RUN_MODE=smoke
fi
case "${BCP_TRAIN_TOPOLOGY:-full}" in
  full) ;;
  idev4_dp2)
    [[ "$BCP_TRAIN_PROFILE" = *_32k_paper_batch ]] && [ "${SMOKE_TEST:-0}" != 1 ] || { echo "idev4_dp2 requires a 32K paper-batch profile" >&2; exit 2; }
    [ "$ALLOCATED_NODE_COUNT" -eq 4 ] || { echo "idev4_dp2 requires exactly 4 allocated nodes" >&2; exit 2; }
    UNUSED_NODES=("${NODES[@]:3}")
    NODES=("${NODES[@]:0:3}")
    export BC_ACTIVE_NODE_COUNT=3 EXPECTED_NUM_NODES=3
    ;;
  *) echo "Unknown BCP_TRAIN_TOPOLOGY: $BCP_TRAIN_TOPOLOGY" >&2; exit 2 ;;
esac
if [ "${PREFLIGHT_ONLY:-0}" != 1 ]; then
  [ "${#NODES[@]}" -eq "$EXPECTED_NUM_NODES" ] || { echo "Requires $EXPECTED_NUM_NODES nodes: search + trainer ranks" >&2; exit 2; }
fi
export TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-deepseek_v4}
export SEARCH_CONDA_ENV=${SEARCH_CONDA_ENV:-cxtgraph}
export SEARCH_CLEAR_LD_PRELOAD=1
export RUN_TAG="qwen35_9b_bcp_${METHOD}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME="${RUN_MODE}_${RUN_TAG}"
export CHECKPOINT_ROOT="$SCRATCH/context-graph-ckpts/$EXPERIMENT_NAME"
RUN_DIR="$PROJECT_ROOT/outputs/$EXPERIMENT_NAME"
mkdir -p logs outputs
mkdir "$RUN_DIR"
exec 3>&2
exec > >(tee -a "$RUN_DIR/suite.log") 2>&1
STAGE=environment_setup
report_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    local message="BCP_RL_FAILED method=$METHOD job=$SLURM_JOB_ID stage=$STAGE exit=$rc artifacts=$RUN_DIR"
    printf '%s\n' "$message"
    printf '%s\n' "$message" | tee -a "$RUN_DIR/failure.log" >&3
  fi
}
trap report_exit EXIT
git rev-parse HEAD | tee "$RUN_DIR/commit.txt"
git diff --exit-code HEAD -- agents envs scripts verl >/dev/null || { echo "Commit tracked code changes before training"; exit 2; }
echo "BC-P RL model=Qwen/Qwen3.5-9B method=$METHOD mode=$RUN_MODE steps=$PLANNED_STEPS nodes=${#NODES[@]} preflight_only=${PREFLIGHT_ONLY:-0}"
echo "Artifacts=$RUN_DIR checkpoints=$CHECKPOINT_ROOT"
echo "Allocation=$ALLOCATED_NODE_COUNT nodes; active=${NODES[*]}; unused=${UNUSED_NODES[*]}"

source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$TRAIN_CONDA_ENV"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
# Keep the retriever's existing /work cache. Trainer loads an absolute snapshot.
export HF_HOME=${SEARCH_HF_HOME:-/work/09281/chc_1996/vista/cache}
export HF_HUB_CACHE=${SEARCH_HF_HUB_CACHE:-$HF_HOME/hub}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export MODEL_PATH="$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION"
export QWEN_TOKENIZER_PATH="$MODEL_PATH"
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/sbsa-linux/lib/stubs:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-$RUN_TAG"
export TRITON_CACHE_DIR="/tmp/triton-$RUN_TAG"
export VLLM_CACHE_ROOT="/tmp/vllm-$RUN_TAG"
TORCH_DEPS=$(find "$CONDA_PREFIX/lib" -path '*/torch/lib/libtorch_global_deps.so' -print -quit)
[ -n "$TORCH_DEPS" ] || { echo "Missing torch global dependencies in $TRAIN_CONDA_ENV"; exit 2; }
export LD_PRELOAD="$TORCH_DEPS${LD_PRELOAD:+:$LD_PRELOAD}"

# Import the actual training stack before spending time loading retrieval data.
STAGE=dependency_preflight
python -u scripts/preflight_bcp_qwen35_rl.py --model-path "$MODEL_PATH" --output "$RUN_DIR/preflight.json"
STAGE=tokenizer_preflight
bash scripts/check_qwen3_observation_tokens.sh
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "BCP_RL_PREFLIGHT_COMPLETE (no training performed)"
  exit 0
fi
if [ -f "${WORK:-/work/09281/chc_1996/vista}/.openai_env" ]; then
  source "${WORK:-/work/09281/chc_1996/vista}/.openai_env"
fi
STAGE=training_setup
[ -n "${OPENAI_API_KEY:-}" ] || { echo "Missing judge credentials"; exit 2; }
export TRAIN_DATA_FILE=data/bc_train.parquet VAL_DATA_FILE=data/bc_test.parquet
sha256sum "$TRAIN_DATA_FILE" "$VAL_DATA_FILE" | tee "$RUN_DIR/data.sha256"
export TOTAL_TRAINING_STEPS=50 TRAINER_VAL_ONLY=False
export TRAINER_RESUME_MODE=disable
export PROMPT_LENGTH=8192 RESPONSE_LENGTH=24576 CONTEXT_LENGTH=32768
export TRAIN_BATCH_SIZE=32 ROLLOUT_N=8 PPO_MINI_BATCH_SIZE=32
# The pinned Qwen3.5 checkpoint has 262144 native text positions. Do not apply
# the older Qwen3-8B base launcher's YaRN override to this model.
export BC_APPLY_YARN=0
if [[ "$BCP_TRAIN_PROFILE" = *_32k_paper_batch ]]; then
  export PPO_MINI_BATCH_SIZE=128
fi
if [ "$BCP_TRAIN_PROFILE" = contextgraph_64k_paper_batch ]; then
  export RESPONSE_LENGTH=57344 CONTEXT_LENGTH=65536 PPO_MINI_BATCH_SIZE=128
fi
export TRAIN_LR=1e-6 USE_KL_LOSS=False ACTOR_KL_LOSS_COEF=0.0 ALGORITHM_KL_COEF=0.0
export CLIP_RATIO_LOW=0.2 CLIP_RATIO_HIGH=0.28
export LORA_RANK=0 ENTROPY_FROM_LOGITS_WITH_CHUNKING=True
export VAL_BEFORE_TRAIN=True TEST_FREQ=10 SAVE_FREQ=5
export TRAIN_MAX_SAMPLES=-1 VAL_MAX_SAMPLES=-1 DATALOADER_NUM_WORKERS=0
export MAX_TURN=100 MAX_SESSION=10 VAL_MAX_SESSION=10 TURN_MAX_NEW_TOKENS=2048
export FINAL_ANSWER_RESERVE=1024 FINAL_ANSWER_SAFETY_MARGIN=64 SESSION_TIMEOUT=3600
export BC_CTXGRAPH_PROTOCOL=controller BC_CONTROLLER_ACTION_POLICY=balanced
export STRUCTURED_MEMORY_ENABLED=0 STRUCTURED_MEMORY_REQUIRED=0
export CONSOLIDATION_INTERVAL=5 AUTO_PRUNE_MAX_ACTIVE=12
if [ "${SMOKE_TEST:-0}" = 1 ]; then
  export TOTAL_TRAINING_STEPS=1
  export TRAIN_BATCH_SIZE=3 ROLLOUT_N=2 PPO_MINI_BATCH_SIZE=3
  export PROMPT_LENGTH=8192 RESPONSE_LENGTH=4096 CONTEXT_LENGTH=12288
  export VAL_BEFORE_TRAIN=False TEST_FREQ=-1 SAVE_FREQ=1
  export TRAIN_MAX_SAMPLES=3 VAL_MAX_SAMPLES=3
  export MAX_TURN=4 MAX_SESSION=1 VAL_MAX_SESSION=1 TURN_MAX_NEW_TOKENS=512 SESSION_TIMEOUT=600
  export BC_DISABLE_WANDB=1
  echo "Smoke: 3 prompts x 2 rollouts, 3 trainer ranks, 1 step, 12K context; not a performance evaluation."
fi
printf '%s\n' "profile=$BCP_TRAIN_PROFILE" "model_revision=$MODEL_REVISION" \
  "steps=$TOTAL_TRAINING_STEPS" "prompt_length=$PROMPT_LENGTH" \
  "response_length=$RESPONSE_LENGTH" "context_length=$CONTEXT_LENGTH" \
  "train_batch_size=$TRAIN_BATCH_SIZE" "rollout_n=$ROLLOUT_N" \
  "ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" "ppo_micro_batch_size_per_gpu=1" \
  "trainer_ranks=$((${#NODES[@]} - 1))" "apply_yarn=$BC_APPLY_YARN" \
  "allocated_nodes=$ALLOCATED_NODE_COUNT" "active_nodes=${NODES[*]}" "unused_nodes=${UNUSED_NODES[*]}" \
  "batch_reference=sunnweiwei/FoldAgent@58a2d6964ecebe99940529eace50a0558901b8a5/scripts/train_bc_qwen3_8b.sh" \
  | tee "$RUN_DIR/training-config.txt"
unset RESUME_CHECKPOINT_PATH RESUME_CHECKPOINT_ROOT QWEN_ENABLE_THINKING
# Native padded HF forward avoids untested Qwen3.5 sequence-packing patches.
ARGS=(
  trainer.resume_mode=disable
  data.seed=42
  actor_rollout_ref.model.use_remove_padding=False
  actor_rollout_ref.model.use_fused_kernels=False
  ++actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.max_model_len="$CONTEXT_LENGTH"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True
  +actor_rollout_ref.rollout.plugin.contextgraph_memory_mode=repaired
  +actor_rollout_ref.rollout.plugin.search_topk_cap=5
  +actor_rollout_ref.rollout.plugin.search_snippet_words=128
  +actor_rollout_ref.rollout.plugin.search_snippet_chars=2000
  +actor_rollout_ref.rollout.plugin.open_page_words=4096
  +actor_rollout_ref.rollout.plugin.open_page_chars=48000
  +actor_rollout_ref.rollout.plugin.apply_chat_template_kwargs.enable_thinking=True
  +actor_rollout_ref.rollout.plugin.apply_chat_template_kwargs.preserve_thinking=True
)
if [ "${SMOKE_TEST:-0}" = 1 ]; then
  ARGS+=(actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]')
fi
if [ "$METHOD" = contextgraph ]; then
  BASE=scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh
else
  BASE=scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh
fi
printf '%s\n' "${ARGS[@]}" > "$RUN_DIR/overrides.txt"
set +e
STAGE=training
bash "$BASE" "${ARGS[@]}"
RC=$?
set -e
echo "BCP_RL_EXIT method=$METHOD exit=$RC checkpoint=$CHECKPOINT_ROOT"
if [ "$RC" -eq 0 ] && [ "${SMOKE_TEST:-0}" = 1 ]; then
  STAGE=checkpoint_audit
  LATEST="$CHECKPOINT_ROOT/latest_checkpointed_iteration.txt"
  [ -s "$LATEST" ] && [ "$(cat "$LATEST")" = 1 ] || { echo "Missing step-1 checkpoint marker"; exit 1; }
  for rank in 0 1 2; do
    for kind in model optim extra_state; do
      SHARD="$CHECKPOINT_ROOT/global_step_1/actor/${kind}_world_size_3_rank_${rank}.pt"
      [ -s "$SHARD" ] || { echo "Missing checkpoint shard: $SHARD"; exit 1; }
    done
  done
  echo "BCP_RL_SMOKE_COMPLETE method=$METHOD steps=1 checkpoint=$CHECKPOINT_ROOT/global_step_1" | tee "$RUN_DIR/smoke-complete.txt"
fi
exit "$RC"
