#!/bin/bash

# Formal one-epoch, full-parameter ContextGraph SFT of the original
# Qwen/Qwen3-8B checkpoint. Run inside an existing four-node Vista idev
# allocation. The final FSDP2 checkpoint is merged into a Hugging Face model.
set -euo pipefail

: "${SCRATCH:?SCRATCH must point to the Vista scratch filesystem}"
PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
FORMAL_DATA_DIR=${FORMAL_DATA_DIR:-$SCRATCH/contextgraph_sft/qwen3_8b_formal_945161_945162}

export TRAIN_FILE=${TRAIN_FILE:-$FORMAL_DATA_DIR/contextgraph_sft_train.parquet}
export VAL_FILE=${VAL_FILE:-$FORMAL_DATA_DIR/contextgraph_sft_validation.parquet}
export MAX_LENGTH=${MAX_LENGTH:-32768}
export TRAIN_BATCH_SIZE=4
export MICRO_BATCH_SIZE=1
export TRAIN_MAX_SAMPLES=-1
export VAL_MAX_SAMPLES=-1
export TOTAL_EPOCHS=1
export SAVE_FREQ=-1
export TRAIN_LR=${TRAIN_LR:-1e-5}
export DATA_PREFLIGHT_ALL=1
export DATA_PREFLIGHT_TIMEOUT=${DATA_PREFLIGHT_TIMEOUT:-1800}

# Do not inherit generic smoke-run paths exported earlier in the same idev
# shell. Formal overrides deliberately use their own variable names.
unset RUN_TAG CHECKPOINT_ROOT MERGED_MODEL_DIR
export RUN_TAG=${FORMAL_RUN_TAG:-${SLURM_JOB_ID:-idev}_qwen3_8b_contextgraph_sft_32k_fullparam}
export CHECKPOINT_ROOT=${FORMAL_CHECKPOINT_ROOT:-$SCRATCH/contextgraph_sft_checkpoints/$RUN_TAG}
export MERGED_MODEL_DIR=${FORMAL_MERGED_MODEL_DIR:-$SCRATCH/contextgraph_sft_models/$RUN_TAG}

for file in "$TRAIN_FILE" "$VAL_FILE"; do
  if [ ! -s "$file" ]; then
    echo "ERROR: formal SFT parquet is missing or empty: $file"
    exit 2
  fi
done
if [ -e "$CHECKPOINT_ROOT" ] && [ -n "$(find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ERROR: checkpoint directory is not empty: $CHECKPOINT_ROOT"
  echo "Set RUN_TAG to a new value or explicitly move the old run aside."
  exit 2
fi
if [ -e "$MERGED_MODEL_DIR" ] && [ -n "$(find "$MERGED_MODEL_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ERROR: merged-model directory is not empty: $MERGED_MODEL_DIR"
  echo "Set RUN_TAG to a new value or explicitly move the old model aside."
  exit 2
fi

set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate cxtgraph
set -u
TRAIN_ROWS=$(python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$TRAIN_FILE').metadata.num_rows)")
VAL_ROWS=$(python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$VAL_FILE').metadata.num_rows)")
if [ "$TRAIN_ROWS" -le 0 ] || [ "$VAL_ROWS" -le 0 ]; then
  echo "ERROR: formal train/validation split must be non-empty"
  exit 2
fi
if [ $((TRAIN_ROWS % TRAIN_BATCH_SIZE)) -ne 0 ]; then
  echo "ERROR: train rows ($TRAIN_ROWS) must be divisible by global batch size ($TRAIN_BATCH_SIZE)"
  exit 2
fi
export TOTAL_TRAINING_STEPS=$((TRAIN_ROWS / TRAIN_BATCH_SIZE))

echo "Formal Qwen3-8B ContextGraph SFT: train_rows=$TRAIN_ROWS val_rows=$VAL_ROWS epochs=$TOTAL_EPOCHS steps=$TOTAL_TRAINING_STEPS max_length=$MAX_LENGTH"
echo "Checkpoint root: $CHECKPOINT_ROOT"
echo "Merged HF model: $MERGED_MODEL_DIR"

bash "$PROJECT_ROOT/scripts/smoke_train_contextgraph_sft_qwen3_8b_4node_idev.sh"

STEP_DIR="$CHECKPOINT_ROOT/global_step_$TOTAL_TRAINING_STEPS"
if [ ! -d "$STEP_DIR" ]; then
  echo "ERROR: final checkpoint is missing: $STEP_DIR"
  exit 3
fi

mkdir -p "$(dirname "$MERGED_MODEL_DIR")"
python -m verl.model_merger merge --backend fsdp --local_dir "$STEP_DIR" --target_dir "$MERGED_MODEL_DIR"
python -c "import glob,json,os; from transformers import AutoConfig,AutoTokenizer; p='$MERGED_MODEL_DIR'; c=AutoConfig.from_pretrained(p,local_files_only=True); AutoTokenizer.from_pretrained(p,local_files_only=True); w=glob.glob(os.path.join(p,'*.safetensors'))+glob.glob(os.path.join(p,'pytorch_model*.bin')); assert getattr(c,'model_type',None)=='qwen3',getattr(c,'model_type',None); assert w,'merged model has no weight files'; print(json.dumps({'merged_model':p,'model_type':c.model_type,'weight_files':len(w)},indent=2))"

printf '%s\n' "$MERGED_MODEL_DIR" > "$CHECKPOINT_ROOT/merged_hf_model_path.txt"
echo "Formal Qwen3-8B ContextGraph SFT completed: $MERGED_MODEL_DIR"
