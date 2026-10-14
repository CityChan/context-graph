#!/bin/bash
# Endpoint runner; the idev launcher prepares the environment and local teacher.
set -euo pipefail
cd "${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}"
: "${RUN_DIR:?Set a unique run directory}"
: "${DEEPSEEK_BASE_URL:?Set local teacher endpoint}"
: "${DEEPSEEK_MODEL:?Set actual served model ID}"
: "${STUDENT_TOKENIZER_PATH:?Set student tokenizer path}"
mkdir -p "$RUN_DIR"
# Run-local metadata prevents concurrent jobs overwriting shared data/ parquets.
if [[ ! -s "$RUN_DIR/data/scienceworld_graph_train.parquet" || ! -s "$RUN_DIR/data/scienceworld_graph_val.parquet" ]]; then
  python scripts/make_scienceworld_data.py --out-dir "$RUN_DIR/data" --n-train 0 --n-val 0 --seed 42
fi
args=(--benchmark scienceworld --data "$RUN_DIR/data/scienceworld_graph_train.parquet" --heldout "$RUN_DIR/data/scienceworld_graph_val.parquet" --output "$RUN_DIR" --tokenizer "$STUDENT_TOKENIZER_PATH" --transport vllm --samples "${SAMPLES:-3}" --candidates "${CANDIDATES:-2}" --workers "${WORKERS:-1}" --start "${START_INDEX:-0}" --seed "${SEED:-42}" --prompt-length "${PROMPT_LENGTH:-8192}" --response-length "${RESPONSE_LENGTH:-24576}" --max-turn "${MAX_TURN:-100}" --max-steps "${MAX_STEPS:-100}" --turn-max-tokens "${TURN_MAX_NEW_TOKENS:-2048}")
[[ ${DRY_RUN:-0} != 1 ]] || args+=(--dry-run)
args+=(--selection "${SFT_SELECTION:-task_balanced}" --teacher-profile "${SFT_TEACHER_PROFILE:-graph_demonstration_v1}" --temperature "${TEACHER_TEMPERATURE:-0.7}")
args+=(--graph-protocol "${SFT_GRAPH_PROTOCOL:-explicit_tools}")
[[ ${TEACHER_PREFLIGHT_ONLY:-0} != 1 ]] || args+=(--preflight-only)
python scripts/generate_bcp_graph_sft.py "${args[@]}"
if [[ ${DRY_RUN:-0} == 1 || ${TEACHER_PREFLIGHT_ONLY:-0} == 1 ]]; then exit 0; fi
python scripts/diagnose_graph_sft.py "$RUN_DIR" --output "$RUN_DIR/diagnostics.json"
SFT_DIR=$(mktemp -d "$RUN_DIR/sft-XXXXXXXX")
rmdir "$SFT_DIR"
BUILDER=scripts/build_bcp_graph_sft.py
if [[ ${SFT_GRAPH_PROTOCOL:-explicit_tools} == controller ]]; then BUILDER=scripts/build_auto_graph_sft.py; fi
python "$BUILDER" "$RUN_DIR" --output "$SFT_DIR" --tokenizer "$STUDENT_TOKENIZER_PATH" --max-length "$(( ${PROMPT_LENGTH:-8192} + ${RESPONSE_LENGTH:-24576} ))" --review
echo "SCIENCEWORLD_GRAPH_SFT_COMPLETE run=$RUN_DIR sft=$SFT_DIR"
