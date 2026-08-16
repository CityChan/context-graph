#!/bin/bash
set -euo pipefail

# Submit the matched Qwen3-30B-A3B-Instruct zero-shot BrowseComp-Plus suite.
# Each method requests eight GH nodes: one retrieval node and seven model nodes.

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
BC_30B_METHODS=${BC_30B_METHODS:-baseline,foldagent,ctxgraph}
FINAL_ANSWER_RESERVE=${FINAL_ANSWER_RESERVE:-1024}

cd "$PROJECT_ROOT"
mkdir -p logs

submit_method() {
  local method=$1
  local script=$2
  local output
  local job_id

  output=$(sbatch --export="ALL,FINAL_ANSWER_RESERVE=$FINAL_ANSWER_RESERVE" "$script")
  job_id=$(printf '%s\n' "$output" | grep -Eo '[0-9]+' | tail -n 1)
  if [ -z "$job_id" ]; then
    echo "ERROR: could not parse job id from: $output" >&2
    exit 1
  fi
  printf '%-12s job=%s script=%s\n' "$method" "$job_id" "$script"
}

method_enabled() {
  case ",$BC_30B_METHODS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

echo "Submitting 30B Instruct BC zero-shot suite: methods=$BC_30B_METHODS final_reserve=$FINAL_ANSWER_RESERVE"

if method_enabled baseline; then
  submit_method baseline scripts/eval_bc_baseline_30b_instruct_8node_zeroshot.sh
fi
if method_enabled foldagent; then
  submit_method foldagent scripts/eval_bc_foldagent_30b_instruct_8node_zeroshot.sh
fi
if method_enabled ctxgraph; then
  submit_method ctxgraph scripts/eval_bc_ctxgraph_30b_instruct_8node_zeroshot.sh
fi

squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.38j %.2t %.10M %.10L %.6D %R"
