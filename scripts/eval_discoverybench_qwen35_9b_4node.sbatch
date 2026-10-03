#!/bin/bash
#SBATCH --job-name=db-9b-paired
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=48:00:00
#SBATCH --account=AST24021
#SBATCH --chdir=/work/09281/chc_1996/vista/context-graph
#SBATCH --output=logs/discoverybench-9b.%j.out
#SBATCH --error=logs/discoverybench-9b.%j.err

set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
cd "$PROJECT_ROOT"
: "${SLURM_JOB_ID:?Submit with sbatch or run inside a four-node allocation}"
: "${SCRATCH:?SCRATCH must be set}"
export DB_CONTEXT_LENGTH=${DB_CONTEXT_LENGTH:-65536} DB_SAMPLES=${DB_SAMPLES:--1}
case "$DB_CONTEXT_LENGTH" in 32768|65536) ;; *) echo 'Expected 32768 or 65536 context'; exit 2;; esac
export DB_AGENT_ENV=${DB_AGENT_ENV:-$SCRATCH/context-graph-discovery/envs/qwen35-api}
export DB_DATA=${DB_DATA:-$PROJECT_ROOT/data/discoverybench_real_test_code.parquet}
export MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export MODEL_PATH=$SCRATCH/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/$MODEL_REVISION
export DISCOVERYBENCH_JUDGE_MODEL=${DISCOVERYBENCH_JUDGE_MODEL:-gpt-5-nano}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT_ROOT"
export SAB_INTERACTIVE_EVAL_FEEDBACK=0 SAB_REAL_EVAL=0
export SAB_PREWARM_PACKAGES=numpy,pandas,scipy,sklearn,statsmodels,xgboost,matplotlib,matplotlib.pyplot,seaborn
export MPLBACKEND=Agg STRUCTURED_MEMORY_ENABLED=0
unset QWEN_ENABLE_THINKING
if [[ -z ${OPENAI_API_KEY:-} && -z ${AZURE_OPENAI_KEY:-} && -z ${AZURE_OPENAI_API_KEY:-} ]]; then
    for credentials in "${WORK:-}/.openai_env" /work/09281/chc_1996/vista/.openai_env "$HOME/.openai_env"; do
        if [[ -f "$credentials" ]]; then set -a; source "$credentials"; set +a; break; fi
    done
fi

agent_preload() {
    local library
    library=$("$DB_AGENT_ENV/bin/python" -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("torch").origin).parent / "lib/libtorch_global_deps.so")')
    [[ -s "$library" ]] || { echo 'Missing Torch TLS library'; exit 2; }
    export LD_PRELOAD="$library"
}

if [[ ${1:-} == _eval ]]; then
    agent_preload
    exec "$DB_AGENT_ENV/bin/python" -u scripts/eval_discoverybench_qwen35.py --method "$2" --endpoint "$3" --output "$DB_RUN_DIR/$2" --data "$DB_DATA" --model-path "$MODEL_PATH" --context-length "$DB_CONTEXT_LENGTH" --samples "$DB_SAMPLES"
fi

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
(( ${#nodes[@]} == 4 )) || { echo 'Expected four nodes: server+ContextGraph, server+FoldAgent'; exit 2; }
mkdir -p "$SCRATCH/context-graph-discovery/runs"
export DB_RUN_DIR=${DB_RUN_DIR:-$(mktemp -d "$SCRATCH/context-graph-discovery/runs/qwen35-9b-paired-${SLURM_JOB_ID}-XXXXXX")}
mkdir -p "$DB_RUN_DIR"
DB_RUN_DIR=$(realpath -e "$DB_RUN_DIR")
export DB_RUN_DIR
exec 8>"$DB_RUN_DIR/launcher.lock"
flock -n 8 || { echo 'This run already has a launcher'; exit 2; }
exec > >(tee -a "$DB_RUN_DIR/suite.log") 2>&1
echo "DISCOVERY_BATCH_START job=$SLURM_JOB_ID context=$DB_CONTEXT_LENGTH samples=$DB_SAMPLES artifacts=$DB_RUN_DIR"

# A private overlay reuses the known-working Qwen agent installation. Scientific
# dependencies are installed here, never into cxtgraph or the SWE environment.
base=${DB_BASE_PYTHON:-/scratch/09281/chc_1996/context-graph-swe/envs/agent-direct-Rs4ngP/bin/python}
mkdir -p "$(dirname "$DB_AGENT_ENV")"
(
    flock -x 7
    if [[ ! -f "$DB_AGENT_ENV/.ready" ]]; then
        [[ -x "$base" ]] || { echo "Missing DB_BASE_PYTHON: $base"; exit 2; }
        "$base" -m venv --system-site-packages "$DB_AGENT_ENV"
        # venv nesting does not automatically inherit the parent venv's packages.
        base_site=$("$base" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        overlay_site=$("$DB_AGENT_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
        printf '%s\n' "$base_site" > "$overlay_site/discovery-parent.pth"
        "$DB_AGENT_ENV/bin/python" -m pip install numpy pandas scipy scikit-learn statsmodels xgboost matplotlib seaborn openpyxl xlrd
        "$DB_AGENT_ENV/bin/python" -m pip freeze --all > "$DB_AGENT_ENV/installed.txt"
        touch "$DB_AGENT_ENV/.ready"
    fi
) 7>"$DB_AGENT_ENV.setup.lock"
cp "$DB_AGENT_ENV/installed.txt" "$DB_RUN_DIR/environment.txt"

# Use the same existing real/test parquet for both agents. If absent, prepare
# once with the existing loader; both evaluations record parquet and file hashes.
if [[ ! -s "$DB_DATA" ]]; then
    mkdir -p "$(dirname "$DB_DATA")"
    (
        flock -x 7
        if [[ ! -s "$DB_DATA" ]]; then
            "$DB_AGENT_ENV/bin/python" scripts/make_discoverybench_data.py --benchmark-dir "$SCRATCH/context-graph-discovery/data" --out-dir "$(dirname "$DB_DATA")" --dataset-type real --split test
        fi
    ) 7>"$DB_DATA.prepare.lock"
fi
agent_preload
"$DB_AGENT_ENV/bin/python" scripts/eval_discoverybench_qwen35.py --preflight --data "$DB_DATA" --samples "$DB_SAMPLES"
unset LD_PRELOAD

pids=()
cleanup() {
    local running pid
    running=" $(jobs -pr | tr '\n' ' ') "
    for pid in "${pids[@]}"; do
        if [[ "$running" == *" $pid "* ]]; then kill -TERM "$pid" 2>/dev/null || true; fi
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
methods=(contextgraph foldagent)
server_pids=()
for pair in 0 1; do
    server=${nodes[$((pair * 2))]}
    method=${methods[$pair]}
    if curl --connect-timeout 3 --max-time 5 -fsS "http://$server:18000/health" >/dev/null 2>&1; then
        echo "Existing model service on $server:18000; refusing to reuse an unowned server"
        exit 2
    fi
    echo "DISCOVERY_PAIR method=$method server=$server evaluator=${nodes[$((pair * 2 + 1))]}"
    srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "$server" env SWE_MODEL_MAX_LEN="$DB_CONTEXT_LENGTH" MODEL_PORT=18000 WORKERS=1 SEED=42 bash scripts/serve_swe_qwen35_9b_vista.sbatch > "$DB_RUN_DIR/server-$method.log" 2>&1 8>&- &
    server_pids+=("$!")
    pids+=("$!")
done
deadline=$((SECONDS + 1200))
while true; do
    ready=0
    for pair in 0 1; do
        kill -0 "${server_pids[$pair]}" 2>/dev/null || { echo "Server ${methods[$pair]} exited; inspect its log"; exit 2; }
        if curl --connect-timeout 3 --max-time 5 -fsS "http://${nodes[$((pair * 2))]}:18000/health" >/dev/null 2>&1; then ready=$((ready + 1)); fi
    done
    (( ready == 2 )) && break
    (( SECONDS < deadline )) || { echo 'Model startup timeout'; exit 2; }
    echo "Waiting for servers: $ready/2"
    sleep 15
done
eval_pids=()
for pair in 0 1; do
    method=${methods[$pair]}
    srun -p gh --jobid="$SLURM_JOB_ID" --overlap -N 1 -n 1 -w "${nodes[$((pair * 2 + 1))]}" bash scripts/eval_discoverybench_qwen35_9b_4node.sbatch _eval "$method" "http://${nodes[$((pair * 2))]}:18000" > "$DB_RUN_DIR/evaluator-$method.log" 2>&1 8>&- &
    eval_pids+=("$!")
    pids+=("$!")
done
status=0
for pid in "${eval_pids[@]}"; do wait "$pid" || status=2; done
echo "DISCOVERY_BATCH_COMPLETE status=$status artifacts=$DB_RUN_DIR"
exit "$status"
