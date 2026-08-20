#!/bin/bash
set -euo pipefail

# Submit matched Qwen3-30B-A3B-Instruct DiscoveryBench zero-shot jobs. Each
# method uses eight GH nodes and the complete 239-query real-test split.

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run this submitter on a Vista login node, not inside an allocation"
  exit 1
fi

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
HF_HOME=${HF_HOME:-/work/09281/chc_1996/vista/cache}
HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
DISCOVERYBENCH_METHODS=${DISCOVERYBENCH_METHODS:-react,fold,ctxgraph}
DISCOVERYBENCH_TIME_LIMIT=${DISCOVERYBENCH_TIME_LIMIT:-12:00:00}
DISCOVERYBENCH_VAL_MAX_SAMPLES=${DISCOVERYBENCH_VAL_MAX_SAMPLES:-239}
DISCOVERYBENCH_VAL_MAX_TURN=${DISCOVERYBENCH_VAL_MAX_TURN:-24}
EVAL_SCRIPT=scripts/eval_discoverybench_qwen3_30b_instruct_8node.sh

cd "$PROJECT_ROOT"
mkdir -p logs

for openai_env in "${WORK:-}/.openai_env" /work/09281/chc_1996/vista/.openai_env "$HOME/.openai_env"; do
  if [ -f "$openai_env" ]; then
    # shellcheck disable=SC1090
    source "$openai_env"
    break
  fi
done

if [ -n "${OPENAI_API_KEY:-}" ] && [ "${OPENAI_API_KEY:-}" != "dummy" ]; then
  :
elif { [ -n "${AZURE_OPENAI_KEY:-}" ] || [ -n "${AZURE_OPENAI_API_KEY:-}" ]; } && [ -n "${AZURE_OPENAI_API_VERSION:-}" ] && [ -n "${AZURE_OPENAI_ENDPOINT:-}" ] && [ -n "${AZURE_OPENAI_DEPLOYMENT_NAME:-}" ]; then
  :
else
  echo "ERROR: DiscoveryBench HMS scoring requires OPENAI_API_KEY or the complete Azure OpenAI credential set"
  exit 2
fi

MODEL_CACHE_DIR="$HF_HUB_CACHE/models--${MODEL_PATH//\//--}"
if ! compgen -G "$MODEL_CACHE_DIR/snapshots/*/config.json" >/dev/null; then
  echo "ERROR: model is not cached at $MODEL_CACHE_DIR"
  echo "Run: hf download $MODEL_PATH"
  exit 3
fi

method_enabled() {
  case ",$DISCOVERYBENCH_METHODS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

data_for_method() {
  case "$1" in
    react) printf '%s\n' data/discoverybench_real_test_code.parquet ;;
    fold) printf '%s\n' data/discoverybench_real_test_code_branch.parquet ;;
    ctxgraph) printf '%s\n' data/discoverybench_real_test_code_graph.parquet ;;
  esac
}

submit_method() {
  local method=$1
  local output
  local job_id

  output=$(sbatch --parsable \
    --job-name="eval-db-${method}-30b-8n" \
    --output="logs/eval-db-${method}-30b-8n.%j.out" \
    --error="logs/eval-db-${method}-30b-8n.%j.err" \
    --time="$DISCOVERYBENCH_TIME_LIMIT" \
    --export="ALL,MODEL_PATH=$MODEL_PATH,DISCOVERYBENCH_METHOD=$method,DISCOVERYBENCH_VAL_MAX_SAMPLES=$DISCOVERYBENCH_VAL_MAX_SAMPLES,DISCOVERYBENCH_VAL_MAX_TURN=$DISCOVERYBENCH_VAL_MAX_TURN" \
    "$EVAL_SCRIPT")
  # Vista prints a submit-validation banner before the parsable job id. Keep
  # only a line that is entirely a Slurm id, optionally followed by ;cluster.
  job_id=$(printf '%s\n' "$output" | sed -nE 's/^([0-9]+)(;[^[:space:]]+)?$/\1/p' | tail -n 1)
  case "$job_id" in
    ''|*[!0-9]*)
      echo "ERROR: could not parse job id from: $output" >&2
      exit 5
      ;;
  esac
  printf '%-10s job=%s nodes=8 limit=%s samples=%s max_turn=%s model=%s\n' \
    "$method" "$job_id" "$DISCOVERYBENCH_TIME_LIMIT" "$DISCOVERYBENCH_VAL_MAX_SAMPLES" "$DISCOVERYBENCH_VAL_MAX_TURN" "$MODEL_PATH"
}

echo "Submitting 30B DiscoveryBench zero-shot suite: model=$MODEL_PATH methods=$DISCOVERYBENCH_METHODS max_turn=$DISCOVERYBENCH_VAL_MAX_TURN"

enabled_method_count=0
for method in react fold ctxgraph; do
  if method_enabled "$method"; then
    data_file=$(data_for_method "$method")
    if [ ! -f "$data_file" ]; then
      echo "ERROR: missing $PROJECT_ROOT/$data_file"
      exit 4
    fi
    enabled_method_count=$((enabled_method_count + 1))
  fi
done
if [ "$enabled_method_count" -eq 0 ]; then
  echo "ERROR: DISCOVERYBENCH_METHODS must enable react, fold, or ctxgraph"
  exit 4
fi

for method in react fold ctxgraph; do
  if method_enabled "$method"; then
    submit_method "$method"
  fi
done

squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.38j %.2t %.10M %.10L %.6D %R"
