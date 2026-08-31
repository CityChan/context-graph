from pathlib import Path


SCRIPT = Path("scripts/smoke_miroverse_controller_sft_deepseek_v4_4node_idev.sh")


def test_smoke_uses_dedicated_deepseek_environment_and_four_nodes():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "CONDA_ENV_NAME=${CONDA_ENV_NAME:-deepseek_v4}" in text
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}" in text
    assert "TEACHER_TP=${TEACHER_TP:-4}" in text
    assert "--tensor-parallel-size $TEACHER_TP" in text
    assert "--tokenizer-mode deepseek_v4" in text
    assert "cxtgraph" not in text


def test_smoke_uses_real_miroverse_and_requires_replay_valid_rows():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "jsonl_sft/MiroVerse-MuSiQue.jsonl" in text
    assert "prepare_miroverse_contextgraph_controller_sft.py" in text
    assert "d['replay_valid'].all()" in text
    assert "all_replay_valid" in text
    assert "contextgraph_controller_smoke.parquet" in text


def test_smoke_has_bounded_mechanics_scope():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "MAX_SAMPLES=${MAX_SAMPLES:-2}" in text
    assert "MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}" in text
    assert "temperature=0.0" not in text  # temperature is controller-owned in Python
