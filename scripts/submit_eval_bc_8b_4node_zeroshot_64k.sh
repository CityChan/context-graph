#!/bin/bash
# Submit the three full 150-example evaluations after all idev smoke tests pass.

set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
BC_METHODS=${BC_METHODS:-baseline,foldagent,contextgraph}
BC_JOB_MODEL_TAG=${BC_JOB_MODEL_TAG:-8b}
BC_EXPERIMENT_MODEL_TAG=${BC_EXPERIMENT_MODEL_TAG:-8b}
BC_EVAL_TIME=${BC_EVAL_TIME:-01:00:00}
BC_CTXGRAPH_PROTOCOL=${BC_CTXGRAPH_PROTOCOL:-legacy}
cd "$PROJECT_ROOT"
mkdir -p logs

method_enabled() {
  case ",$BC_METHODS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

echo "Submitting BC-P eval: model=$MODEL_PATH methods=$BC_METHODS nodes=4 time=$BC_EVAL_TIME"
echo "ContextGraph protocol: $BC_CTXGRAPH_PROTOCOL"

for method in baseline foldagent contextgraph; do
  if ! method_enabled "$method"; then
    continue
  fi
  method_protocol=legacy
  if [ "$method" = "contextgraph" ]; then method_protocol=$BC_CTXGRAPH_PROTOCOL; fi
  job_id=$(MODEL_PATH="$MODEL_PATH" \
    BC_METHOD="$method" \
    BC_CTXGRAPH_PROTOCOL="$method_protocol" \
    BC_EXPERIMENT_MODEL_TAG="$BC_EXPERIMENT_MODEL_TAG" \
    BC_CONTEXT_LENGTH=65536 \
    BC_PROMPT_LENGTH=8192 \
    BC_RESPONSE_LENGTH=57344 \
    BC_YARN_FACTOR=2.0 \
    BC_YARN_ORIGINAL_LENGTH=32768 \
    BC_FINAL_ANSWER_RESERVE=1024 \
    BC_VAL_MAX_SAMPLES=-1 \
    sbatch --parsable \
    --job-name="eval-bc-${BC_JOB_MODEL_TAG}-${method}-${method_protocol}-64k" \
    --output="logs/eval-bc-${BC_JOB_MODEL_TAG}-${method}-${method_protocol}-64k.%j.out" \
    --error="logs/eval-bc-${BC_JOB_MODEL_TAG}-${method}-${method_protocol}-64k.%j.err" \
    --nodes=4 \
    --time="$BC_EVAL_TIME" \
    scripts/eval_bc_baseline_8b_4node_zeroshot.sh)
  echo "$method: submitted job $job_id"
done
