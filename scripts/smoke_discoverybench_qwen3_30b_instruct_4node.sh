#!/bin/bash
set -euo pipefail

# Run inside a 4-node GH idev allocation. Select react, fold, or ctxgraph with
# DISCOVERYBENCH_METHOD; all three use the same official test queries.
DISCOVERYBENCH_METHOD=${DISCOVERYBENCH_METHOD:-react}
case "$DISCOVERYBENCH_METHOD" in
  react)
    DISCOVERYBENCH_DATA_FILE=data/discoverybench_real_test_code.parquet
    ;;
  fold)
    DISCOVERYBENCH_DATA_FILE=data/discoverybench_real_test_code_branch.parquet
    ;;
  ctxgraph)
    DISCOVERYBENCH_DATA_FILE=data/discoverybench_real_test_code_graph.parquet
    ;;
  *)
    echo "ERROR: DISCOVERYBENCH_METHOD must be react, fold, or ctxgraph"
    exit 1
    ;;
esac

export CODE_BENCHMARK_LABEL=DiscoveryBench
export CODE_BENCHMARK_PROFILE=discoverybench
export CODE_BENCHMARK_DATA_FILE=$DISCOVERYBENCH_DATA_FILE
export CODE_BENCHMARK_TRAIN_MODULE=scripts.train_discoverybench
export CODE_BENCHMARK_PREPARE_HINT="Run: python scripts/make_discoverybench_data.py --out-dir data --dataset-type real --split test"
export SAB_METHOD=$DISCOVERYBENCH_METHOD
export EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}
export SAB_RUN_TAG=${SAB_RUN_TAG:-smoke}
export SAB_REAL_EVAL=0
export DISCOVERYBENCH_REAL_EVAL=${DISCOVERYBENCH_REAL_EVAL:-1}
export SAB_DUMP_VALIDATION=${SAB_DUMP_VALIDATION:-1}
export SAB_VAL_MAX_SAMPLES=${DISCOVERYBENCH_VAL_MAX_SAMPLES:-1}
export SAB_TRAIN_MAX_SAMPLES=${DISCOVERYBENCH_TRAIN_MAX_SAMPLES:-4}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export SAB_PROMPT_LENGTH=${DISCOVERYBENCH_PROMPT_LENGTH:-16384}
export SAB_RESPONSE_LENGTH=${DISCOVERYBENCH_RESPONSE_LENGTH:-8192}
export SAB_MAX_TOKEN_LEN_PER_GPU=${DISCOVERYBENCH_MAX_TOKEN_LEN_PER_GPU:-24576}
export SAB_VAL_MAX_TURN=${DISCOVERYBENCH_VAL_MAX_TURN:-12}
export SAB_TURN_MAX_NEW_TOKENS=${DISCOVERYBENCH_TURN_MAX_NEW_TOKENS:-1024}
# DiscoveryBench primarily uses the core tabular/statistical stack. Avoid
# eagerly importing SAB-only chemistry/biology packages such as deepchem and
# DeepPurpose: their native TensorFlow initialization can deadlock a Ray agent
# worker after vLLM has initialized CUDA. Tasks can still import them on demand.
export SAB_PREWARM_PACKAGES=${DISCOVERYBENCH_PREWARM_PACKAGES:-numpy,pandas,scipy,sklearn,statsmodels,xgboost,matplotlib,matplotlib.pyplot,seaborn}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-eval_${DISCOVERYBENCH_METHOD}_discoverybench_30b_4n_${SAB_RUN_TAG}_${SLURM_JOB_ID:-idev}}

exec bash scripts/eval_sab_react_30b_instruct_8node_smoke.sh
