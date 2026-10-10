#!/usr/bin/env bash
# Run within the intended training environment/allocation. No environment
# replacement, Ray shutdown, node allocation, or background service management.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the intended training checkpoint}"
PYTHON=${PYTHON:-python}
RUN_DIR=${RUN_DIR:-output/scienceworld-graphrpo-v3-long-$(date +%Y%m%d-%H%M%S)-$$}
exec "$PYTHON" -m scripts.train_scienceworld_graphrpo \
  "actor_rollout_ref.model.path=$MODEL_PATH" \
  "data.train_files=${TRAIN_FILE:-data/scienceworld_graph_train.parquet}" \
  "data.val_files=${VAL_FILE:-data/scienceworld_graph_val.parquet}" \
  "trainer.default_local_dir=$RUN_DIR" \
  "trainer.n_gpus_per_node=${GPUS_PER_NODE:-1}" \
  "trainer.nnodes=${NNODES:-1}" \
  "scienceworld_profile_prepare_only=${DRY_RUN:-false}" "$@"
