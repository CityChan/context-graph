#!/bin/bash
set -euo pipefail

# Submit matched Qwen3-8B DiscoveryBench zero-shot jobs. Each method uses four
# GH nodes and evaluates the same complete 239-query real-test split by default.

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
DISCOVERYBENCH_METHODS=${DISCOVERYBENCH_METHODS:-react,fold,ctxgraph}
DISCOVERYBENCH_TIME_LIMIT=${DISCOVERYBENCH_TIME_LIMIT:-08:00:00}
DISCOVERYBENCH_VAL_MAX_SAMPLES=${DISCOVERYBENCH_VAL_MAX_SAMPLES:-239}
EVAL_SCRIPT=scripts/eval_discoverybench_qwen3_8b_4node.sh

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
  local output
  local job_id

  output=$(sbatch \
    --job-name="eval-db-${method}-8b-4n" \
    --output="logs/eval-db-${method}-8b-4n.%j.out" \
    --error="logs/eval-db-${method}-8b-4n.%j.err" \
    --time="$DISCOVERYBENCH_TIME_LIMIT" \
    --export="ALL,DISCOVERYBENCH_METHOD=$method,DISCOVERYBENCH_VAL_MAX_SAMPLES=$DISCOVERYBENCH_VAL_MAX_SAMPLES" \
    "$EVAL_SCRIPT")
  job_id=$(printf '%s\n' "$output" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$job_id" ]; then
    echo "ERROR: could not parse job id from: $output" >&2
    exit 1
  fi
  printf '%-10s job=%s nodes=4 limit=%s samples=%s\n' \
    "$method" "$job_id" "$DISCOVERYBENCH_TIME_LIMIT" "$DISCOVERYBENCH_VAL_MAX_SAMPLES"
}

echo "Submitting Qwen3-8B DiscoveryBench zero-shot suite: methods=$DISCOVERYBENCH_METHODS"

for method in react fold ctxgraph; do
  if method_enabled "$method"; then
    submit_method "$method"
  fi
done

squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.38j %.2t %.10M %.10L %.6D %R"
