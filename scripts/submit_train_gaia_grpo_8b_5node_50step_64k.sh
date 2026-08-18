#!/bin/bash
# Submit matched 64K GRPO training on the GAIA train split for three methods.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
CHECKPOINT_BASE=${CHECKPOINT_BASE:-${SCRATCH:-/scratch/09281/chc_1996}/context-graph-ckpts}
GAIA_TRAIN_TIME=${GAIA_TRAIN_TIME:-48:00:00}
GAIA_TRAIN_NUM_NODES=${GAIA_TRAIN_NUM_NODES:-5}
GAIA_TRAIN_STEPS=${GAIA_TRAIN_STEPS:-50}
GAIA_TRAIN_METHODS=${GAIA_TRAIN_METHODS:-baseline,foldagent,ctxgraph}
GAIA_CONTEXT_TAG=${GAIA_CONTEXT_TAG:-64k}
GAIA_PROMPT_LENGTH=${GAIA_PROMPT_LENGTH:-8192}
GAIA_RESPONSE_LENGTH=${GAIA_RESPONSE_LENGTH:-57344}
GAIA_CONTEXT_LENGTH=${GAIA_CONTEXT_LENGTH:-65536}
GAIA_FINAL_ANSWER_RESERVE=${GAIA_FINAL_ANSWER_RESERVE:-2048}
GAIA_FINAL_ANSWER_SAFETY_MARGIN=${GAIA_FINAL_ANSWER_SAFETY_MARGIN:-64}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
GAIA_PREPARE_DEV_SPLIT_IF_MISSING=${GAIA_PREPARE_DEV_SPLIT_IF_MISSING:-1}
GAIA_DATA_PYTHON=${GAIA_DATA_PYTHON:-/work/09281/chc_1996/vista/miniconda3/envs/cxtgraph/bin/python}

cd "$PROJECT_ROOT"
mkdir -p logs

if [ -n "${WORK:-}" ] && [ -f "$WORK/.openai_env" ]; then
  # shellcheck disable=SC1090
  source "$WORK/.openai_env"
fi
if [ -z "${OPENAI_API_KEY:-}" ] || [ "${OPENAI_API_KEY:-}" = "dummy" ]; then
  echo "ERROR: GAIA GRPO training requires OPENAI_API_KEY for answer judging" >&2
  exit 1
fi

if [ ! -s data/gaia_train.parquet ] || [ ! -s data/gaia_train_branch.parquet ] || [ ! -s data/gaia_train_graph.parquet ] || [ ! -s data/gaia_holdout.parquet ] || [ ! -s data/gaia_holdout_branch.parquet ] || [ ! -s data/gaia_holdout_graph.parquet ]; then
  if [ "$GAIA_PREPARE_DEV_SPLIT_IF_MISSING" != "1" ]; then
    echo "ERROR: GAIA train/holdout parquets are missing and GAIA_PREPARE_DEV_SPLIT_IF_MISSING=$GAIA_PREPARE_DEV_SPLIT_IF_MISSING" >&2
    exit 2
  fi
  if [ ! -x "$GAIA_DATA_PYTHON" ]; then
    echo "ERROR: GAIA data Python is not executable: $GAIA_DATA_PYTHON" >&2
    exit 2
  fi
  echo "Creating matched 80/20 train/holdout splits from public GAIA validation data"
  "$GAIA_DATA_PYTHON" scripts/split_gaia_validation_for_training.py --input-dir data --output-dir data --holdout-fraction 0.2 --seed 42
fi

for data_file in \
  data/gaia_train.parquet \
  data/gaia_train_branch.parquet \
  data/gaia_train_graph.parquet \
  data/gaia_holdout.parquet \
  data/gaia_holdout_branch.parquet \
  data/gaia_holdout_graph.parquet; do
  if [ ! -s "$data_file" ]; then
    echo "ERROR: missing or empty GAIA parquet: $PROJECT_ROOT/$data_file" >&2
    echo "       Rebuild with: python scripts/split_gaia_validation_for_training.py --input-dir data --output-dir data" >&2
    exit 3
  fi
done

method_enabled() {
  case ",$GAIA_TRAIN_METHODS," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

submit_method() {
  local method=$1
  local batch_script=$2
  local train_data=$3
  local val_data=$4
  local short_method=$5
  local experiment_name="train_gaia_${method}_grpo_8b_5n_50s_${GAIA_CONTEXT_TAG}_devsplit_${STAMP}"
  local checkpoint_root="$CHECKPOINT_BASE/$experiment_name"
  local job_name="train-gaia-${short_method}-8b-${GAIA_CONTEXT_TAG}"
  local export_vars
  local job_id

  export_vars="ALL,EXPECTED_NUM_NODES=$GAIA_TRAIN_NUM_NODES,EXPERIMENT_NAME=$experiment_name,CHECKPOINT_ROOT=$checkpoint_root,RUN_TAG=gaia_grpo_5n_50step_${GAIA_CONTEXT_TAG},TRAIN_DATA_FILE=$train_data,VAL_DATA_FILE=$val_data,TRAINER_VAL_ONLY=False,VAL_BEFORE_TRAIN=True,TOTAL_TRAINING_STEPS=$GAIA_TRAIN_STEPS,TEST_FREQ=10,SAVE_FREQ=10,PROMPT_LENGTH=$GAIA_PROMPT_LENGTH,RESPONSE_LENGTH=$GAIA_RESPONSE_LENGTH,CONTEXT_LENGTH=$GAIA_CONTEXT_LENGTH,BC_YARN_FACTOR=2.0,BC_YARN_ORIGINAL_LENGTH=32768,TRAIN_BATCH_SIZE=32,PPO_MINI_BATCH_SIZE=32,ROLLOUT_N=8,TRAIN_LR=1e-6,USE_KL_LOSS=False,ACTOR_KL_LOSS_COEF=0.0,ALGORITHM_KL_COEF=0.0,CLIP_RATIO_LOW=0.2,CLIP_RATIO_HIGH=0.28,MAX_TURN=100,MAX_SESSION=10,VAL_MAX_SESSION=10,TURN_MAX_NEW_TOKENS=2048,FINAL_ANSWER_RESERVE=$GAIA_FINAL_ANSWER_RESERVE,FINAL_ANSWER_SAFETY_MARGIN=$GAIA_FINAL_ANSWER_SAFETY_MARGIN,SESSION_TIMEOUT=3600,BC_SEARCH_TIMEOUT_SECONDS=600,ENTROPY_FROM_LOGITS_WITH_CHUNKING=True"
  job_id=$(sbatch --parsable \
    --job-name="$job_name" \
    --nodes="$GAIA_TRAIN_NUM_NODES" \
    --time="$GAIA_TRAIN_TIME" \
    --output="logs/${job_name}.%j.out" \
    --error="logs/${job_name}.%j.err" \
    --export="$export_vars" \
    "$batch_script")
  printf '%-12s job=%s train=%s val=%s checkpoint=%s\n' "$method" "$job_id" "$train_data" "$val_data" "$checkpoint_root"
}

echo "Submitting matched GAIA GRPO training: methods=$GAIA_TRAIN_METHODS nodes=$GAIA_TRAIN_NUM_NODES steps=$GAIA_TRAIN_STEPS prompt=$GAIA_PROMPT_LENGTH response=$GAIA_RESPONSE_LENGTH context=$GAIA_CONTEXT_LENGTH final_reserve=$GAIA_FINAL_ANSWER_RESERVE safety_margin=$GAIA_FINAL_ANSWER_SAFETY_MARGIN"

if method_enabled baseline; then
  submit_method baseline scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh data/gaia_train.parquet data/gaia_holdout.parquet base
fi
if method_enabled foldagent; then
  submit_method foldagent scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh data/gaia_train_branch.parquet data/gaia_holdout_branch.parquet fold
fi
if method_enabled ctxgraph; then
  submit_method ctxgraph scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh data/gaia_train_graph.parquet data/gaia_holdout_graph.parquet cg
fi

squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.32j %.2t %.10M %.10L %.6D %R"
