from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_train_contextgraph_sft_qwen3_8b_20k_4node_idev.sh"


def test_20k_smoke_uses_the_four_longest_formal_samples():
    text = SCRIPT.read_text()
    assert "contextgraph_sft_longest4.parquet" in text
    assert "export MAX_LENGTH=20480" in text
    assert "export TRAIN_MAX_SAMPLES=4" in text
    assert "export TOTAL_TRAINING_STEPS=1" in text
    assert "smoke_train_contextgraph_sft_qwen3_8b_4node_idev.sh" in text
