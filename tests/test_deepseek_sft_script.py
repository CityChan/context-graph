from pathlib import Path


SCRIPT = Path("scripts/generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh")
INTERACTIVE_SCRIPT = Path("scripts/generate_ctxgraph_sft_deepseek_v4_interactive_8node.sh")
IDEV4_SCRIPT = Path("scripts/smoke_interactive_ctxgraph_deepseek_v4_4node_idev.sh")


def test_deepseek_sft_script_uses_open_checkpoint_and_scratch():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "deepseek-ai/DeepSeek-V4-Flash-0731" in text
    assert "SCRATCH" in text
    assert "/work/09281/chc_1996/vista/cache/hub/models--deepseek-ai" not in text
    assert "require_scratch_path MODEL_PATH" in text


def test_deepseek_sft_script_requires_v4_serving_features():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "vLLM >= 0.25.0" in text
    assert "vllm serve --help=all" in text
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


def test_deepseek_interactive_script_covers_both_train_domains_and_strict_trace():
    text = INTERACTIVE_SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1" in text
    assert "alfworld_graph" in text
    assert "scienceworld_graph" in text
    assert "deepseek-ai/DeepSeek-V4-Flash-0731" in text
    assert '--reasoning-effort "$REASONING_EFFORT"' in text
    assert "--require-graph-trace" in text
    assert "--min-graph-quality-score 1.0" in text
    assert "--max-invalid-graph-ops 0" in text
    assert "_train.parquet" in text
    assert "require_scratch_path MODEL_PATH" in text
    assert "SHARED_HF_HOME" not in text
    assert "SHARED_HF_HUB_CACHE" not in text


def test_deepseek_four_node_idev_smoke_is_conservative():
    text = IDEV4_SCRIPT.read_text(encoding="utf-8")
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-4}" in text
    assert "TEACHER_TP=${TEACHER_TP:-4}" in text
    assert "MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}" in text
    assert "MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}" in text
    assert "REASONING_EFFORT=${REASONING_EFFORT:-non-thinking}" in text
    assert "generate_ctxgraph_sft_deepseek_v4_interactive_8node.sh" in text
