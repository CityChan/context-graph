from pathlib import Path


SCRIPT = Path("scripts/smoke_generate_ctxgraph_sft_qwen3_6_27b_4node_idev.sh")


def test_qwen36_smoke_uses_two_nodes_from_an_idev_allocation():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Qwen/Qwen3.6-27B" in text
    assert 'if [ "$NUM_NODES" -lt 2 ]' in text
    assert "SEARCH_NODE=${NODELIST[0]}" in text
    assert "TEACHER_NODE=${NODELIST[1]}" in text
    assert "--tensor-parallel-size 1" in text
    assert "#SBATCH" not in text
    assert "ray start" not in text


def test_qwen36_smoke_keeps_executor_filters_and_outputs_sft():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "MAX_SAMPLES=${MAX_SAMPLES:-2}" in text
    assert "--save-messages" in text
    assert "scripts/build_contextgraph_sft.py" in text
    assert "--min-task-reward 1.0" in text
    assert "--min-structural-graph-ops 1" in text
    assert "--max-invalid-graph-ops 0" in text
    assert "contextgraph_sft_train.parquet" in text


def test_qwen36_smoke_uses_supported_text_only_vllm_mode():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "vLLM >= 0.19.0" in text
    assert "vllm serve --help=all" in text
    assert "--reasoning-parser qwen3" in text
    assert "--language-model-only" in text
    assert "QWEN_ENABLE_THINKING=True" in text
