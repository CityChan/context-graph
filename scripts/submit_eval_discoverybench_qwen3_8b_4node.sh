#!/bin/bash
set -euo pipefail

# Submit matched Qwen3-8B DiscoveryBench zero-shot jobs. Each method uses four
# GH nodes and evaluates the same complete 239-query real-test split by default.

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
DISCOVERYBENCH_METHODS=${DISCOVERYBENCH_METHODS:-react,fold,ctxgraph}
DISCOVERYBENCH_TIME_LIMIT=${DISCOVERYBENCH_TIME_LIMIT:-02:00:00}
DISCOVERYBENCH_VAL_MAX_SAMPLES=${DISCOVERYBENCH_VAL_MAX_SAMPLES:-239}
DISCOVERYBENCH_CTXGRAPH_PROTOCOL=${DISCOVERYBENCH_CTXGRAPH_PROTOCOL:-controller}
DISCOVERYBENCH_CONTROLLER_ACTION_POLICY=${DISCOVERYBENCH_CONTROLLER_ACTION_POLICY:-structural}
DISCOVERYBENCH_JOB_MODEL_TAG=${DISCOVERYBENCH_JOB_MODEL_TAG:-8b}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-cxtgraph}
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
ROLLOUT_LOAD_FORMAT=${ROLLOUT_LOAD_FORMAT:-dummy}
ROLLOUT_LAYERED_SUMMON=${ROLLOUT_LAYERED_SUMMON:-False}
DRY_RUN=${DRY_RUN:-0}
EVAL_SCRIPT=scripts/eval_discoverybench_qwen3_8b_4node.sh

case "$DISCOVERYBENCH_CTXGRAPH_PROTOCOL" in
  legacy|controller) ;;
  *)
    echo "ERROR: DISCOVERYBENCH_CTXGRAPH_PROTOCOL must be legacy or controller; got $DISCOVERYBENCH_CTXGRAPH_PROTOCOL" >&2
    exit 2
    ;;
esac
case "$DISCOVERYBENCH_CONTROLLER_ACTION_POLICY" in
  balanced|structural) ;;
  *)
    echo "ERROR: DISCOVERYBENCH_CONTROLLER_ACTION_POLICY must be balanced or structural; got $DISCOVERYBENCH_CONTROLLER_ACTION_POLICY" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
mkdir -p logs

method_enabled() {
  case ",$DISCOVERYBENCH_METHODS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

submit_method() {
  local method=$1
  local method_protocol=legacy
  if [ "$method" = "ctxgraph" ]; then
    method_protocol=$DISCOVERYBENCH_CTXGRAPH_PROTOCOL
  fi
  local output
  local job_id

  local export_vars="ALL,MODEL_PATH=$MODEL_PATH,CONDA_ENV_NAME=$CONDA_ENV_NAME,LORA_ADAPTER_PATH=$LORA_ADAPTER_PATH,LORA_RANK=$LORA_RANK,LORA_ALPHA=$LORA_ALPHA,ROLLOUT_LOAD_FORMAT=$ROLLOUT_LOAD_FORMAT,ROLLOUT_LAYERED_SUMMON=$ROLLOUT_LAYERED_SUMMON,DISCOVERYBENCH_METHOD=$method,SAB_CTXGRAPH_PROTOCOL=$method_protocol,SAB_CONTROLLER_ACTION_POLICY=$DISCOVERYBENCH_CONTROLLER_ACTION_POLICY,DISCOVERYBENCH_VAL_MAX_SAMPLES=$DISCOVERYBENCH_VAL_MAX_SAMPLES"
  local -a submit_args=(sbatch \
    --job-name="eval-db-${method}-${method_protocol}-${DISCOVERYBENCH_JOB_MODEL_TAG}-4n" \
    --output="logs/eval-db-${method}-${method_protocol}-${DISCOVERYBENCH_JOB_MODEL_TAG}-4n.%j.out" \
    --error="logs/eval-db-${method}-${method_protocol}-${DISCOVERYBENCH_JOB_MODEL_TAG}-4n.%j.err" \
    --time="$DISCOVERYBENCH_TIME_LIMIT" \
    --export="$export_vars" \
    "$EVAL_SCRIPT")
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY_RUN'
    printf ' %q' "${submit_args[@]}"
    printf '\n'
    return
  fi
  output=$("${submit_args[@]}")
  job_id=$(printf '%s\n' "$output" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$job_id" ]; then
    echo "ERROR: could not parse job id from: $output" >&2
    exit 1
  fi
  printf '%-10s job=%s protocol=%s nodes=4 limit=%s samples=%s\n' \
    "$method" "$job_id" "$method_protocol" "$DISCOVERYBENCH_TIME_LIMIT" "$DISCOVERYBENCH_VAL_MAX_SAMPLES"
}

echo "Submitting Qwen3-8B DiscoveryBench zero-shot suite: methods=$DISCOVERYBENCH_METHODS"
echo "ContextGraph protocol: $DISCOVERYBENCH_CTXGRAPH_PROTOCOL action_policy=$DISCOVERYBENCH_CONTROLLER_ACTION_POLICY"

for method in react fold ctxgraph; do
  if method_enabled "$method"; then
    submit_method "$method"
  fi
done

if [ "$DRY_RUN" = "0" ]; then
  squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.38j %.2t %.10M %.10L %.6D %R"
fi
