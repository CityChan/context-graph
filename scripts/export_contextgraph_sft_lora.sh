#!/bin/bash
# Export a PEFT adapter from a ContextGraph FSDP SFT checkpoint.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/work/09281/chc_1996/vista/context-graph}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${SCRATCH:-/scratch/09281/chc_1996}/contextgraph_sft_checkpoints/947686_qwen36_27b_sft_full/global_step_147}
EXPORT_ROOT=${EXPORT_ROOT:-${SCRATCH:-/scratch/09281/chc_1996}/contextgraph_sft_exports/qwen36_27b_scienceworld_step147_v2}
LORA_ALPHA=${LORA_ALPHA:-64}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-deepseek_v4}
ADAPTER_DIR=$EXPORT_ROOT/lora_adapter

case "$LORA_ALPHA" in
  ''|*[!0-9]*) echo "ERROR: LORA_ALPHA must be a positive integer" >&2; exit 2 ;;
  0) echo "ERROR: LORA_ALPHA must be greater than zero" >&2; exit 2 ;;
esac

for required in fsdp_config.json huggingface/config.json; do
  if [ ! -s "$CHECKPOINT_DIR/$required" ]; then
    echo "ERROR: missing checkpoint file: $CHECKPOINT_DIR/$required" >&2
    exit 2
  fi
done

if [ -e "$EXPORT_ROOT" ]; then
  echo "ERROR: EXPORT_ROOT already exists; choose a new empty path: $EXPORT_ROOT" >&2
  exit 2
fi

mkdir -p "$(dirname "$EXPORT_ROOT")"
cd "$PROJECT_ROOT"
set +u
source /work/09281/chc_1996/vista/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"
set -u

python -m verl.model_merger merge --backend fsdp --local_dir "$CHECKPOINT_DIR" --target_dir "$EXPORT_ROOT" --lora-adapter-only --lora-alpha "$LORA_ALPHA" --use_cpu_initialization --trust-remote-code

for required in adapter_config.json adapter_model.safetensors; do
  if [ ! -s "$ADAPTER_DIR/$required" ]; then
    echo "ERROR: adapter export did not create $ADAPTER_DIR/$required" >&2
    exit 3
  fi
done

python -c 'import json,sys; from safetensors import safe_open; path,expected=sys.argv[1],int(sys.argv[2]); config=json.load(open(path+"/adapter_config.json",encoding="utf-8")); keys=list(safe_open(path+"/adapter_model.safetensors",framework="pt",device="cpu").keys()); legacy=[key for key in keys if key.startswith("base_model.model.model.layers.")]; assert config.get("lora_alpha")==expected,config; assert int(config.get("r",0))>0,config; assert keys,"adapter has no tensors"; assert not legacy,f"legacy Qwen3.5 LoRA keys remain: {legacy[:3]}"; print(json.dumps({"adapter":path,"r":config["r"],"lora_alpha":config["lora_alpha"],"tensor_count":len(keys)}))' "$ADAPTER_DIR" "$LORA_ALPHA"

echo "ContextGraph SFT LoRA export passed: $ADAPTER_DIR"
