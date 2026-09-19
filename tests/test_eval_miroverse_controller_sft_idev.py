from pathlib import Path


SCRIPT = Path("scripts/eval_miroverse_controller_sft_idev.sh")


def test_idev_wrapper_uses_node_local_caches_and_eager_mode():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "TORCHINDUCTOR_CACHE_DIR" in text
    assert "TRITON_CACHE_DIR" in text
    assert "VLLM_CACHE_ROOT" in text
    assert "export CC=gcc" in text
    assert "export CXX=g++" in text
    assert "export CUDAHOSTCXX=g++" in text
    assert "--enforce-eager" in text
    assert "GUIDED_DECODING=${GUIDED_DECODING:-0}" in text
    assert "DECODING_FLAG=--guided-decoding" in text
    assert "DECODING_FLAG=--no-guided-decoding" in text
    assert "-$DECODING_TAG.log" in text
    assert "contextgraph_sft_validation.parquet" in text
