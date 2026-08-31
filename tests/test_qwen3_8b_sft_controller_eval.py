from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "eval_bc_baseline_8b_4node_zeroshot.sh"
WRAPPER = ROOT / "scripts" / "eval_bc_contextgraph_qwen3_8b_sft_4node_idev.sh"


def test_browsecomp_runner_accepts_a_local_merged_hf_model():
    text = RUNNER.read_text(encoding="utf-8")
    assert 'if [ -d "$MODEL_PATH" ]' in text
    assert 'TRAINER_CACHE_DIR="$MODEL_PATH"' in text
    assert "local MODEL_PATH is missing config.json" in text
    assert "local MODEL_PATH has no non-empty Hugging Face weight files" in text


def test_sft_eval_matches_aug27_controller_protocol():
    text = WRAPPER.read_text(encoding="utf-8")
    assert "unset MODEL_PATH EXPERIMENT_NAME" in text
    assert "contextgraph_sft_models" in text
    assert "export BC_METHOD=contextgraph" in text
    assert "export BC_CTXGRAPH_PROTOCOL=controller" in text
    assert "export BC_CONTROLLER_ACTION_POLICY=structural" in text
    assert "export BC_CONTEXT_LENGTH=65536" in text
    assert "export BC_PROMPT_LENGTH=8192" in text
    assert "export BC_RESPONSE_LENGTH=57344" in text
    assert "export BC_VAL_MAX_SAMPLES=-1" in text
    assert "export BC_DISABLE_WANDB=1" in text
    assert "eval_bc_baseline_8b_4node_zeroshot.sh" in text
