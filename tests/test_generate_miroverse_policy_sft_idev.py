from pathlib import Path


SCRIPT = Path("scripts/generate_miroverse_policy_sft_idev.sh")


def test_policy_generation_wrapper_uses_full_converter_and_strict_preflight():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text
    assert "inspect_miroverse_policy_source.py" in text
    assert "prepare_miroverse_contextgraph_policy_sft.py" in text
    assert "contextgraph_sft_train.parquet" in text
    assert "contextgraph_sft_validation.parquet" in text
    assert "--all-samples" in text
    assert "RUN_PREFLIGHT=${RUN_PREFLIGHT:-0}" in text
