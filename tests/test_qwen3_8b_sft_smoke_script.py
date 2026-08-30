from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_train_contextgraph_sft_qwen3_8b_4node_idev.sh"
GENERIC_SCRIPT = ROOT / "scripts" / "smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh"


def test_qwen3_8b_smoke_is_full_parameter_and_ignores_stale_model_path():
    text = SCRIPT.read_text()
    assert "unset MODEL_PATH" in text
    assert "MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}" in text
    assert "TRAIN_CONDA_ENV=${TRAIN_CONDA_ENV:-cxtgraph}" in text
    assert "export LORA_RANK=0" in text
    assert "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}" in text
    assert "models--Qwen--Qwen3-8B/snapshots" in text
    assert "export ALLOW_WORK_MODEL_CACHE=1" in text


def test_qwen3_8b_smoke_uses_only_the_selected_production_shard():
    text = SCRIPT.read_text()
    assert "945161_0/alfworld/contextgraph_sft_train.parquet" in text
    assert "945161_0/alfworld/contextgraph_sft_validation.parquet" in text
    assert "TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-4}" in text
    assert "VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-4}" in text


def test_generic_smoke_reports_the_selected_model_and_separate_validation_data():
    text = GENERIC_SCRIPT.read_text()
    assert 'echo "$MODEL_ID ContextGraph SFT training smoke"' in text
    assert 'data.val_files="$VAL_FILES"' in text
    assert 'TRAIN_FILES="[$TRAIN_FILE]"' in text
    assert 'VAL_FILES="[$VAL_FILE]"' in text
    assert '"${WORK_MODEL_CACHE_ROOT%/}"/*' in text
