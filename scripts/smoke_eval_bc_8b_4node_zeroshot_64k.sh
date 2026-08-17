#!/bin/bash
# Run this inside an existing 4-node Vista GH idev allocation.

set -euo pipefail

PROJECT_ROOT=/work/09281/chc_1996/vista/context-graph
cd "$PROJECT_ROOT"

# 64K means total model context, not 64K response on top of the prompt.
export BC_CONTEXT_LENGTH=${BC_CONTEXT_LENGTH:-65536}
export BC_PROMPT_LENGTH=${BC_PROMPT_LENGTH:-8192}
export BC_RESPONSE_LENGTH=${BC_RESPONSE_LENGTH:-$((BC_CONTEXT_LENGTH - BC_PROMPT_LENGTH))}
export BC_YARN_FACTOR=${BC_YARN_FACTOR:-2.0}
export BC_YARN_ORIGINAL_LENGTH=${BC_YARN_ORIGINAL_LENGTH:-32768}
export BC_FINAL_ANSWER_RESERVE=${BC_FINAL_ANSWER_RESERVE:-1024}
export BC_VAL_MAX_SAMPLES=${BC_VAL_MAX_SAMPLES:-8}
export BC_DISABLE_WANDB=${BC_DISABLE_WANDB:-1}

METHODS=${BC_METHODS:-"baseline foldagent contextgraph"}

echo "=============================================================="
echo "  BrowseComp-Plus 8B zero-shot 64K smoke evaluation"
echo "  methods: $METHODS"
echo "  total context: $BC_CONTEXT_LENGTH"
echo "  prompt/response: $BC_PROMPT_LENGTH/$BC_RESPONSE_LENGTH"
echo "  final-answer reserve: $BC_FINAL_ANSWER_RESERVE"
echo "  validation samples per method: $BC_VAL_MAX_SAMPLES"
echo "=============================================================="

for method in $METHODS; do
  echo ""
  echo "########## SMOKE START $method ##########"
  BC_METHOD="$method" bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh
  echo "########## SMOKE DONE  $method ##########"
done
