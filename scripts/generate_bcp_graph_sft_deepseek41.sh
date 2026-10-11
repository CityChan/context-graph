#!/bin/bash
#SBATCH -J bcp-ds41-sft
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 04:00:00
#SBATCH -A AST24021
# One GPU for BC-P retrieval; teacher is an API or an already-running vLLM.
set -euo pipefail
cd "${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}"
source "${CONDA_SH:-/work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh}"
conda activate cxtgraph
: "${SCRATCH:?Set SCRATCH}"
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
TOKENIZER=${STUDENT_TOKENIZER_PATH:-Qwen/Qwen3.5-9B}
RUN_DIR=${RUN_DIR:-$SCRATCH/contextgraph_sft/bcp_deepseek41/${SLURM_JOB_ID:-local}-$(date +%Y%m%d_%H%M%S)}
TRANSPORT=${TEACHER_TRANSPORT:-deepseek}
SEARCH_PORT=${SEARCH_PORT:-18999}
SEARCH_PID=
if [[ "$TRANSPORT" == vllm ]]; then
    : "${DEEPSEEK_BASE_URL:?Set the existing V4.1 vLLM base URL}"
    : "${DEEPSEEK_MODEL:?Set the exact served V4.1 model ID}"
fi
if [[ ${DRY_RUN:-0} != 1 && "$TRANSPORT" == deepseek ]]; then
    : "${DEEPSEEK_API_KEY:?Set the teacher API key in the environment}"
fi
mkdir -p "$RUN_DIR"
echo "BCP_GRAPH_SFT run=$RUN_DIR teacher=${DEEPSEEK_MODEL:-deepseek-flash} transport=$TRANSPORT"
cleanup() {
    if [[ -n "$SEARCH_PID" ]]; then
        kill "$SEARCH_PID" 2>/dev/null || true
        wait "$SEARCH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
if [[ -z ${LOCAL_SEARCH_URL:-} ]]; then
    export LOCAL_SEARCH_URL="http://127.0.0.1:$SEARCH_PORT"
    if [[ ${DRY_RUN:-0} != 1 ]]; then
        if curl --noproxy '*' -fsS "$LOCAL_SEARCH_URL/health" >/dev/null 2>&1; then
            echo "Port $SEARCH_PORT is occupied; explicitly set LOCAL_SEARCH_URL to reuse a retrieval service."
            exit 2
        fi
        SEARCH_HF_HOME=${SEARCH_HF_HOME:-/work/09281/chc_1996/vista/cache}
        HF_HOME="$SEARCH_HF_HOME" HF_HUB_CACHE="$SEARCH_HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SEARCH_CORPUS_CACHE_READ_ONLY=1 NUM_GPUS=1 MAX_BATCH_SIZE=64 python -u envs/search_server.py --model Qwen/Qwen3-Embedding-8B --host 127.0.0.1 --port "$SEARCH_PORT" --corpus Tevatron/browsecomp-plus-corpus --corpus-embedding-dataset miaolu3/browsecomp-plus >"$RUN_DIR/search.log" 2>&1 &
        SEARCH_PID=$!
        ready=0
        for ((i=0; i<900; i++)); do
            if curl --noproxy '*' -fsS "$LOCAL_SEARCH_URL/health" >/dev/null 2>&1; then ready=1; break; fi
            if ! kill -0 "$SEARCH_PID" 2>/dev/null; then tail -n 40 "$RUN_DIR/search.log"; exit 2; fi
            sleep 1
        done
        [[ "$ready" == 1 ]] || { echo "Retrieval startup timed out: $RUN_DIR/search.log"; exit 2; }
    fi
fi
args=(--data "${DATA_PATH:-data/bc_train.parquet}" --heldout "${HELDOUT_PATH:-data/bc_test.parquet}" --output "$RUN_DIR" --tokenizer "$TOKENIZER" --transport "$TRANSPORT" --samples "${SAMPLES:-25}" --start "${START_INDEX:-0}" --candidates "${CANDIDATES:-2}" --workers "${WORKERS:-2}" --seed "${SEED:-42}")
[[ ${DRY_RUN:-0} != 1 ]] || args+=(--dry-run)
[[ ${PREFLIGHT_ONLY:-0} != 1 ]] || args+=(--preflight-only)
python scripts/generate_bcp_graph_sft.py "${args[@]}"
if [[ ${DRY_RUN:-0} == 1 || ${PREFLIGHT_ONLY:-0} == 1 ]]; then exit 0; fi
# Curation uses a new directory, so resume never overwrites a previous dataset.
SFT_DIR=$(mktemp -d "$RUN_DIR/sft-XXXXXXXX")
rmdir "$SFT_DIR"
python scripts/build_bcp_graph_sft.py "$RUN_DIR" --output "$SFT_DIR" --tokenizer "$TOKENIZER" --review
echo "BCP_GRAPH_SFT_COMPLETE run=$RUN_DIR sft=$SFT_DIR"
