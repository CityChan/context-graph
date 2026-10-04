#!/usr/bin/env bash
#SBATCH --job-name=gram-rl-smoke
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/gram-rl-smoke.%j.out
#SBATCH --error=logs/gram-rl-smoke.%j.err
# Also accepts direct bash execution in an idle four-node idev allocation.
set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
: "${SLURM_JOB_ID:?Use sbatch or an idle four-node idev allocation}"
: "${SCRATCH:?SCRATCH must be set}"
SELF=scripts/smoke_gram_rl_qwen35_9b_4node.sbatch
export CONDA_SH=${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}
# Match the current Qwen3.5 BC-P training environment; no package installation.
export TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-deepseek_v4}
export MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export MODEL_PATH="$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION"
export PYTHONPATH="$PROJECT_ROOT" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME="$SCRATCH/hf_cache" HF_HUB_CACHE="$SCRATCH/hf_cache/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false RAY_DEDUP_LOGS=0
export GRAM_BENCHMARK=${GRAM_BENCHMARK:-document-stream}
case "$GRAM_BENCHMARK" in
    document-stream) export GRAM_TRAIN_RANKS=3; helper_index=0; head_index=1 ;;
    bcp) export GRAM_TRAIN_RANKS=2; helper_index=1; head_index=2 ;;
    *) echo 'Unsupported GRAM_BENCHMARK'; exit 2 ;;
esac

training_env() {
    set +u
    source "$CONDA_SH"
    conda activate "$TRAIN_CONDA_ENV"
    set -u
    export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8
    export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$CUDA_HOME/targets/sbsa-linux/lib/stubs:${LIBRARY_PATH:-}"
    export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
    export NCCL_HOSTID="${SLURMD_NODENAME:-$(hostname -s)}" FLASHINFER_WORKSPACE_BASE=/tmp
    local library
    library=$(find "$CONDA_PREFIX/lib" -path '*/torch/lib/libtorch_global_deps.so' -print -quit)
    [[ -s "$library" ]] || { echo 'Missing Torch global dependencies'; exit 2; }
    export LD_PRELOAD="$library"
    export TORCHINDUCTOR_CACHE_DIR="/tmp/gram-rl-$SLURM_JOB_ID/inductor"
    export TRITON_CACHE_DIR="/tmp/gram-rl-$SLURM_JOB_ID/triton"
    export VLLM_CACHE_ROOT="/tmp/gram-rl-$SLURM_JOB_ID/vllm"
}

gpu_idle() {
    local processes
    processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    [[ -z "${processes//[[:space:]]/}" ]] || { echo "GPU busy on $(hostname -s); use idle nodes"; exit 2; }
    if pgrep -u "$UID" -x raylet >/dev/null; then echo 'Existing Ray instance; use idle nodes'; exit 2; fi
}

case "${1:-}" in
    _idle) gpu_idle; exit 0 ;;
    _prepare)
        training_env
        python -u scripts/preflight_bcp_qwen35_rl.py --model-path "$MODEL_PATH" --output "$GRAM_RUN_DIR/preflight.json"
        python -m scripts.gram_rl_smoke prepare --run "$GRAM_RUN_DIR" --benchmark "$GRAM_BENCHMARK"
        exit 0 ;;
    _ray-head|_ray-worker)
        training_env
        exec 9>"/tmp/contextgraph-swe-server-${UID}.lock"
        flock -n 9 || { echo 'GPU service lock is occupied'; exit 2; }
        gpu_idle
        node_ip=$(getent ahostsv4 "$(hostname -s)" | awk 'NR==1 {print $1; exit}')
        [[ -n "$node_ip" ]] || exit 2
        if [[ "$1" == _ray-head ]]; then
            exec ray start --head --node-ip-address="$node_ip" --port="$GRAM_RAY_PORT" --num-cpus=70 --num-gpus=1 --include-dashboard=false --block
        fi
        exec ray start --address="$RAY_ADDRESS" --node-ip-address="$node_ip" --num-cpus=70 --num-gpus=1 --block ;;
    _ready)
        training_env
        exec python -m scripts.gram_rl_smoke ray-ready --run "$GRAM_RUN_DIR" --world-size "$GRAM_TRAIN_RANKS" ;;
    _train)
        training_env
        export GRAM_TRAIN_DATA="$GRAM_RUN_DIR/data/train/data.parquet"
        export GRAM_VAL_DATA="$GRAM_RUN_DIR/data/validation/data.parquet"
        export GRAM_MEMORY_MODEL=Qwen/Qwen3.5-9B GRAM_MEMORY_REVISION="$MODEL_REVISION" GRAM_FROZEN_MEMORY=true
        export GRAM_NNODES="$GRAM_TRAIN_RANKS" GRAM_GPUS_PER_NODE=1 GRAM_TP=1 GRAM_TRAINING_STEPS=2
        export GRAM_CONTEXT_LENGTH=12288 GRAM_STEP_TOKENS=1024 GRAM_EPISODE_TOKENS=8192 GRAM_MAX_STEPS=16
        args=(
            data.train_batch_size=1 data.val_batch_size=1 data.dataloader_num_workers=0 data.shuffle=False
            actor_rollout_ref.rollout.n=3 actor_rollout_ref.actor.ppo_mini_batch_size=1
            actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.use_fused_kernels=False
            ++actor_rollout_ref.model.override_config.attn_implementation=sdpa
            actor_rollout_ref.actor.entropy_from_logits_with_chunking=True actor_rollout_ref.actor.entropy_checkpointing=True
            actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False
            actor_rollout_ref.ref.fsdp_config.param_offload=True
            actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.max_model_len=12288
            actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.agent.num_workers=3
            actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=False
            actor_rollout_ref.rollout.val_kwargs.temperature=0.0
            +actor_rollout_ref.rollout.engine_kwargs.vllm.language_model_only=True
            +actor_rollout_ref.rollout.plugin.apply_chat_template_kwargs.enable_thinking=False
            trainer.total_epochs=2 trainer.val_before_train=False trainer.test_freq=2 trainer.save_freq=2
            trainer.resume_mode=disable trainer.default_local_dir="$GRAM_RUN_DIR/checkpoints"
            trainer.experiment_name="gram-rl-smoke-$SLURM_JOB_ID"
            actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]'
            +ray_kwargs.ray_init.runtime_env.env_vars.GRAM_TRACE_DIR="$GRAM_TRACE_DIR"
        )
        if [[ "$GRAM_BENCHMARK" == bcp ]]; then
            args+=(data.train_batch_size=2 data.val_batch_size=2
                actor_rollout_ref.rollout.n=2 actor_rollout_ref.actor.ppo_mini_batch_size=2
                actor_rollout_ref.rollout.agent.num_workers=2
                +actor_rollout_ref.rollout.plugin.gram.benchmark=bcp
                +ray_kwargs.ray_init.runtime_env.env_vars.JUDGE_MODEL="$JUDGE_MODEL")
        fi
        printf '%s\n' "${args[@]}" > "$GRAM_RUN_DIR/overrides.txt"
        bash scripts/train_gram.sh "${args[@]}" 2>&1 | tee "$GRAM_RUN_DIR/trainer.log"
        python -m scripts.gram_rl_smoke audit --run "$GRAM_RUN_DIR" --world-size "$GRAM_TRAIN_RANKS" --benchmark "$GRAM_BENCHMARK"
        echo "GRAM_RL_SMOKE_COMPLETE artifacts=$GRAM_RUN_DIR" | tee "$GRAM_RUN_DIR/smoke-complete.txt"
        exit 0 ;;
    '') ;;
    *) echo "Usage: sbatch $SELF, or bash $SELF inside idle four-node idev"; exit 2 ;;
esac

mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?Missing node list}")
(( ${#nodes[@]} == 4 )) || { echo "Expected exactly four nodes for GRAM $GRAM_BENCHMARK smoke"; exit 2; }
mkdir -p "$SCRATCH/context-graph-gram/runs"
export GRAM_RUN_DIR=$(mktemp -d "$SCRATCH/context-graph-gram/runs/gram-rl-smoke-${SLURM_JOB_ID}-XXXXXX")
export GRAM_TRACE_DIR="$GRAM_RUN_DIR/traces"
mkdir -p "$GRAM_TRACE_DIR"
exec > >(tee -a "$GRAM_RUN_DIR/suite.log") 2>&1
git rev-parse HEAD > "$GRAM_RUN_DIR/code_commit.txt"
git diff --binary HEAD > "$GRAM_RUN_DIR/working_tree.patch"
printf '%s\n' "${nodes[@]}" > "$GRAM_RUN_DIR/nodes.txt"
echo "GRAM_RL_SMOKE_START benchmark=$GRAM_BENCHMARK helper=${nodes[$helper_index]} trainer_ranks=$GRAM_TRAIN_RANKS artifacts=$GRAM_RUN_DIR"
if [[ "$GRAM_BENCHMARK" == bcp ]]; then
    echo 'BC-P RL: four training questions, two test questions for validation; two updates, batch=2 rollout=2.'
    if [[ -z ${OPENAI_API_KEY:-} || ${OPENAI_API_KEY:-} == dummy ]]; then
        for credentials in "${WORK:-/work/09281/chc_1996/vista}/.openai_env" /work/09281/chc_1996/vista/.openai_env "$HOME/.openai_env"; do
            if [[ -f "$credentials" ]]; then source "$credentials"; fi
            if [[ -n ${OPENAI_API_KEY:-} && ${OPENAI_API_KEY:-} != dummy ]]; then break; fi
        done
    fi
    [[ -n ${OPENAI_API_KEY:-} && ${OPENAI_API_KEY:-} != dummy ]] || { echo 'Missing BC-P judge credentials'; exit 2; }
    export OPENAI_API_KEY JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano}
    if [[ -n ${JUDGE_BASE_URL:-} ]]; then export OPENAI_BASE_URL="$JUDGE_BASE_URL"; else unset OPENAI_BASE_URL; fi
    unset OPENAI_URL
    common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
    data_root="$PROJECT_ROOT/data"
    if [[ ! -s "$data_root/bc_train.parquet" && -n "$common_dir" ]]; then data_root="$common_dir/../data"; fi
    export GRAM_BCP_TRAIN_SOURCE=${GRAM_BCP_TRAIN_SOURCE:-$data_root/bc_train.parquet}
    export GRAM_BCP_VAL_SOURCE=${GRAM_BCP_VAL_SOURCE:-$data_root/bc_test.parquet}
    for source in "$GRAM_BCP_TRAIN_SOURCE" "$GRAM_BCP_VAL_SOURCE"; do
        [[ -s "$source" ]] || { echo "Missing BC-P source: $source"; exit 2; }
    done
    export LOCAL_SEARCH_URL="http://${nodes[0]}:18999" BENCHMARK=bcp MODEL_ID=Qwen/Qwen3.5-9B
else
    echo 'Two optimizer updates; HotpotQA dev row 0 for updates, row 1 for validation; batch=1 rollout=3.'
fi
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,${nodes[0]},${nodes[1]},${nodes[2]},${nodes[3]}"
export no_proxy="$NO_PROXY"
echo "Training environment=$TRAIN_CONDA_ENV; context=12288; full-parameter GRPO with KL."
pids=()
cleanup() {
    local live pid
    live=" $(jobs -pr | tr '\n' ' ') "
    for pid in "${pids[@]}"; do
        if [[ "$live" == *" $pid "* ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for node in "${nodes[@]}"; do
    srun --overlap -N 1 -n 1 -w "$node" bash "$SELF" _idle
done
srun --overlap -N 1 -n 1 -w "${nodes[$head_index]}" bash "$SELF" _prepare > "$GRAM_RUN_DIR/preparation.log" 2>&1 || { cat "$GRAM_RUN_DIR/preparation.log"; exit 2; }
if [[ "$GRAM_BENCHMARK" == bcp ]]; then
    srun --overlap -N 1 -n 1 -w "${nodes[0]}" env -u LD_PRELOAD SEARCH_PORT=18999 AGENT_CONDA_ENV=cxtgraph bash scripts/eval_bcp_qwen38_4node_idev.sh _search > "$GRAM_RUN_DIR/search.log" 2>&1 &
    search_pid=$!; pids+=("$search_pid")
    deadline=$((SECONDS + 1800))
    until curl --connect-timeout 3 --max-time 5 -fsS "$LOCAL_SEARCH_URL/health" >/dev/null 2>&1; do
        kill -0 "$search_pid" 2>/dev/null || { echo 'BC-P retriever exited; inspect search.log'; exit 2; }
        (( SECONDS < deadline )) || { echo 'BC-P retriever startup timeout'; exit 2; }
        echo 'Waiting for BC-P retriever'; sleep 10
    done
fi
export GRAM_MEMORY_ENDPOINT="http://${nodes[$helper_index]}:18000"
srun --overlap -N 1 -n 1 -w "${nodes[$helper_index]}" env -u LD_PRELOAD SWE_MODEL_MAX_LEN=32768 MODEL_PORT=18000 WORKERS=1 SEED=42 bash scripts/serve_swe_qwen35_9b_vista.sbatch > "$GRAM_RUN_DIR/memory-server.log" 2>&1 &
helper_pid=$!; pids+=("$helper_pid")
deadline=$((SECONDS + 1200))
until curl --connect-timeout 3 --max-time 5 -fsS "$GRAM_MEMORY_ENDPOINT/health" >/dev/null 2>&1; do
    kill -0 "$helper_pid" 2>/dev/null || { echo 'Memory server exited; inspect memory-server.log'; exit 2; }
    (( SECONDS < deadline )) || { echo 'Memory server startup timeout'; exit 2; }
    echo 'Waiting for frozen memory helper'; sleep 10
done
export GRAM_RAY_PORT=$((16379 + SLURM_JOB_ID % 10000))
head_ip=$(getent ahostsv4 "${nodes[$head_index]}" | awk 'NR==1 {print $1; exit}')
[[ -n "$head_ip" ]] || { echo 'Cannot resolve trainer head'; exit 2; }
export RAY_ADDRESS="$head_ip:$GRAM_RAY_PORT"
for ((index=head_index; index<4; index++)); do
    mode=_ray-worker
    if (( index == head_index )); then mode=_ray-head; fi
    srun --overlap -N 1 -n 1 -w "${nodes[$index]}" bash "$SELF" "$mode" > "$GRAM_RUN_DIR/ray-$index.log" 2>&1 &
    pids+=("$!")
    sleep 5
done
deadline=$((SECONDS + 300))
until srun --overlap -N 1 -n 1 -w "${nodes[$head_index]}" timeout 30 bash "$SELF" _ready > "$GRAM_RUN_DIR/ray-ready.log" 2>&1; do
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null || { echo 'Owned helper/Ray process exited'; exit 2; }; done
    (( SECONDS < deadline )) || { cat "$GRAM_RUN_DIR/ray-ready.log"; exit 2; }
    echo "Waiting for $GRAM_TRAIN_RANKS trainer GPUs"; sleep 5
done
srun --overlap -N 1 -n 1 -w "${nodes[$head_index]}" bash "$SELF" _train
