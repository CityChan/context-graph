#!/bin/bash

set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <checkpoint_root> <hf_repo_id> <experiment_name> [private:true|false]"
  exit 2
fi

CHECKPOINT_ROOT=$1
HF_REPO_ID=$2
EXPERIMENT_NAME=$3
HF_REPO_PRIVATE=${4:-false}

export HF_HUB_OFFLINE=0
unset TRANSFORMERS_OFFLINE

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found in PATH"
  exit 3
fi

echo "--- Hugging Face auth ---"
hf auth whoami

if [ ! -d "$CHECKPOINT_ROOT" ]; then
  echo "Checkpoint root does not exist: $CHECKPOINT_ROOT"
  exit 4
fi

LATEST_FILE="${CHECKPOINT_ROOT}/latest_checkpointed_iteration.txt"
LATEST_STEP=""
if [ -s "$LATEST_FILE" ]; then
  LATEST_STEP=$(tr -cd '0-9' < "$LATEST_FILE" || true)
fi

LATEST_CKPT=""
if [ -n "$LATEST_STEP" ] && [ -d "${CHECKPOINT_ROOT}/global_step_${LATEST_STEP}" ]; then
  LATEST_CKPT="${CHECKPOINT_ROOT}/global_step_${LATEST_STEP}"
else
  LATEST_CKPT=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)
fi

if [ -z "$LATEST_CKPT" ] || [ ! -d "$LATEST_CKPT" ]; then
  echo "No global_step_* checkpoint found under $CHECKPOINT_ROOT"
  exit 5
fi

STEP_NAME=$(basename "$LATEST_CKPT")
HF_PRIVATE_ARGS=()
case "$HF_REPO_PRIVATE" in
  true|True|TRUE|1|yes|YES)
    HF_PRIVATE_ARGS=(--private)
    ;;
esac

echo "--- Uploading checkpoint ---"
echo "Local: $LATEST_CKPT"
echo "Repo:  $HF_REPO_ID"
echo "Path:  ${EXPERIMENT_NAME}/${STEP_NAME}"

hf upload "$HF_REPO_ID" "$LATEST_CKPT" "${EXPERIMENT_NAME}/${STEP_NAME}" \
  --repo-type model \
  "${HF_PRIVATE_ARGS[@]}" \
  --commit-message "Upload ${EXPERIMENT_NAME} ${STEP_NAME}"

if [ -s "$LATEST_FILE" ]; then
  hf upload "$HF_REPO_ID" "$LATEST_FILE" "${EXPERIMENT_NAME}/latest_checkpointed_iteration.txt" \
    --repo-type model \
    "${HF_PRIVATE_ARGS[@]}" \
    --commit-message "Update ${EXPERIMENT_NAME} latest checkpoint"
fi

echo "HF checkpoint sync completed: ${HF_REPO_ID}/${EXPERIMENT_NAME}/${STEP_NAME}"
