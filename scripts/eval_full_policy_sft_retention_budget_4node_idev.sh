#!/bin/bash

# High-budget, retention-first evaluation of the full-policy Qwen3-8B SFT
# checkpoint on BC-P, GAIA, and the fixed ALFWorld split. Run inside an
# existing four- or five-node Vista idev allocation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/09281/chc_1996}
RUN_BC=${RUN_BC:-1}
RUN_GAIA=${RUN_GAIA:-1}
RUN_ALFWORLD=${RUN_ALFWORLD:-1}
FULL_POLICY_SFT_MODEL_PATH=${FULL_POLICY_SFT_MODEL_PATH:-$SCRATCH_ROOT/contextgraph_sft_models/miroverse_full_policy_qwen3_8b_bs16_1ep_v1_step174_hf}

case "$RUN_BC:$RUN_GAIA:$RUN_ALFWORLD" in
  0:0:0|0:0:1|0:1:0|0:1:1|1:0:0|1:0:1|1:1:0|1:1:1) ;;
  *) echo "ERROR: RUN_BC, RUN_GAIA, and RUN_ALFWORLD must each be 0 or 1" >&2; exit 2 ;;
esac

test -s "$FULL_POLICY_SFT_MODEL_PATH/config.json" || { echo "ERROR: invalid full-policy SFT model: $FULL_POLICY_SFT_MODEL_PATH" >&2; exit 2; }

cd "$PROJECT_ROOT"
mkdir -p logs
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
SUITE_LOG=${SUITE_LOG:-logs/full_policy_sft_retention_budget.${SLURM_JOB_ID:-nojob}.${STAMP}.log}

echo "=============================================================="
echo "  Full-policy SFT retention-first high-budget evaluation"
echo "  Model: $FULL_POLICY_SFT_MODEL_PATH"
echo "  BC-P: 64K, 150 turns, consolidation/20, auto-prune/48"
echo "  GAIA: 64K, 150 turns, consolidation/20, auto-prune/48"
echo "  ALFWorld: 32K, 120 turns, consolidation/20, auto-prune/48"
echo "  Suite log: $SUITE_LOG"
echo "=============================================================="

set +e
(
  if [ "$RUN_BC" = "1" ] || [ "$RUN_GAIA" = "1" ]; then
    RUN_BC="$RUN_BC" RUN_GAIA="$RUN_GAIA" STAMP="$STAMP" \
    EVAL_VARIANT=full_policy_sft_retention EVAL_MODEL_TAG=miroverse_full_policy_step174 \
    EVAL_MODEL_PATH="$FULL_POLICY_SFT_MODEL_PATH" EVAL_CTXGRAPH_PROTOCOL=full_policy \
    BC_CONTEXT_LENGTH=65536 BC_PROMPT_LENGTH=8192 BC_RESPONSE_LENGTH=57344 \
    BC_MAX_TURN=150 BC_MAX_SESSION=15 BC_FINAL_ANSWER_RESERVE=2048 \
    BC_CONSOLIDATION_INTERVAL=20 BC_AUTO_PRUNE_MAX_ACTIVE=48 \
    GAIA_CONTEXT_LENGTH=65536 GAIA_PROMPT_LENGTH=8192 GAIA_RESPONSE_LENGTH=57344 \
    GAIA_MAX_TURN=150 GAIA_MAX_SESSION=15 GAIA_FINAL_ANSWER_RESERVE=2048 \
    GAIA_CONSOLIDATION_INTERVAL=20 GAIA_AUTO_PRUNE_MAX_ACTIVE=48 \
    bash scripts/eval_bc_gaia_qwen3_8b_base_idev.sh
  fi

  if [ "$RUN_ALFWORLD" = "1" ]; then
    FULL_POLICY_SFT_MODEL_PATH="$FULL_POLICY_SFT_MODEL_PATH" \
    ALFWORLD_EXPERIMENT_PREFIX=full_policy_sft_retention \
    ALFWORLD_PROMPT_LENGTH=4096 ALFWORLD_RESPONSE_LENGTH=28672 \
    ALFWORLD_MAX_TOKEN_LEN_PER_GPU=32768 \
    ALFWORLD_MAX_TURN=120 ALFWORLD_VAL_MAX_TURN=120 \
    ALFWORLD_TURN_MAX_NEW_TOKENS=256 ALFWORLD_BRANCH_LEN=4096 \
    ALFWORLD_MAX_SESSION=6 ALFWORLD_VAL_MAX_SESSION=6 \
    ALFWORLD_SESSION_TIMEOUT=1200 ALFWORLD_CONSOLIDATION_INTERVAL=20 \
    ALFWORLD_AUTO_PRUNE_MAX_ACTIVE=48 ALFWORLD_FINAL_ANSWER_RESERVE=2048 \
    bash scripts/eval_alfworld_ctxgraph_qwen3_8b_full_policy_sft_4node_idev.sh
  fi
) 2>&1 | tee "$SUITE_LOG"
RC=${PIPESTATUS[0]}
set -e

echo "Retention-first evaluation log: $SUITE_LOG"
exit "$RC"
