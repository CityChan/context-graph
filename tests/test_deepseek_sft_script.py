from pathlib import Path


SCRIPT = Path("scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh")


def test_deepseek_sft_script_uses_open_checkpoint_and_scratch():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "deepseek-ai/DeepSeek-V4-Flash-0731" in text
    assert "SCRATCH" in text
    assert "/work/09281/chc_1996/vista/cache/hub/models--deepseek-ai" not in text


def test_deepseek_sft_script_requires_v4_serving_features():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "vLLM >= 0.25.0" in text
    assert "--tokenizer-mode deepseek_v4" in text
    assert "--enable-expert-parallel" in text
    assert "--tensor-parallel-size $TEACHER_TP" in text
    assert "REASONING_EFFORT=${REASONING_EFFORT:-non-thinking}" in text
    assert 'if [ "$PREFLIGHT_ONLY" = "1" ]' in text


def test_deepseek_sft_script_keeps_eval_splits_out_by_default():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "ALLOW_EVAL_DATA=${ALLOW_EVAL_DATA:-0}" in text
    assert "refusing evaluation split" in text
    assert "--min-structural-graph-ops 1" in text
