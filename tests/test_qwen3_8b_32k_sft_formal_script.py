from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_contextgraph_sft_qwen3_8b_32k_4node_idev.sh"
GENERIC_SCRIPT = ROOT / "scripts" / "smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh"


def test_formal_32k_sft_uses_full_curated_split_and_full_parameters():
    text = SCRIPT.read_text()
    assert "contextgraph_sft_train.parquet" in text
    assert "contextgraph_sft_validation.parquet" in text
    assert "export MAX_LENGTH=${MAX_LENGTH:-32768}" in text
    assert "export TRAIN_MAX_SAMPLES=-1" in text
    assert "export VAL_MAX_SAMPLES=-1" in text
    assert "export TOTAL_EPOCHS=1" in text
    assert "export TRAIN_BATCH_SIZE=4" in text
    assert "smoke_train_contextgraph_sft_qwen3_8b_4node_idev.sh" in text
    assert "unset RUN_TAG CHECKPOINT_ROOT MERGED_MODEL_DIR" in text
    assert "FORMAL_RUN_TAG" in text
    assert "FORMAL_CHECKPOINT_ROOT" in text
    assert "FORMAL_MERGED_MODEL_DIR" in text
    assert "export DATA_PREFLIGHT_ALL=1" in text
    assert "DATA_PREFLIGHT_TIMEOUT=${DATA_PREFLIGHT_TIMEOUT:-1800}" in text


def test_formal_32k_sft_derives_steps_and_merges_final_checkpoint():
    text = SCRIPT.read_text()
    assert "TOTAL_TRAINING_STEPS=$((TRAIN_ROWS / TRAIN_BATCH_SIZE))" in text
    assert "python -m verl.model_merger merge --backend fsdp" in text
    assert "merged_hf_model_path.txt" in text
    assert 'SAVE_FREQ=${SAVE_FREQ:-1}' in GENERIC_SCRIPT.read_text()


def test_miroverse_formal_wrapper_uses_4k_four_nodes_and_twelve_hours():
    wrapper = (ROOT / "scripts" / "train_miroverse_controller_sft_qwen3_8b_4k_4node_12h.sh").read_text()
    assert "#SBATCH -N 4" in wrapper
    assert "#SBATCH -t 12:00:00" in wrapper
    assert "miroverse_controller_qwen3_8b" in wrapper
    assert "export MAX_LENGTH=4096" in wrapper
    assert "train_contextgraph_sft_qwen3_8b_32k_4node_idev.sh" in wrapper
