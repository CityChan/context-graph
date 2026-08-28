from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_model_merger_supports_adapter_only_export_with_explicit_alpha():
    source = _read("verl/model_merger/base_model_merger.py")
    assert '"--lora-adapter-only"' in source
    assert '"--lora-alpha"' in source
    assert "self.config.lora_alpha or 0" in source
    assert "Checkpoint contains no LoRA parameters" in source


def test_export_script_keeps_only_a_verified_lora_adapter():
    source = _read("scripts/export_contextgraph_sft_lora.sh")
    assert "--lora-adapter-only" in source
    assert '--lora-alpha "$LORA_ALPHA"' in source
    assert "adapter_model.safetensors" in source
    assert "EXPORT_ROOT already exists" in source


def test_all_three_eval_runners_accept_the_same_lora_overrides():
    runners = (
        "scripts/eval_bc_baseline_8b_4node_zeroshot.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
        "scripts/eval_sab_react_30b_instruct_8node_smoke.sh",
    )
    for path in runners:
        source = _read(path)
        assert 'LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-}' in source
        assert 'actor_rollout_ref.model.lora_adapter_path="$LORA_ADAPTER_PATH"' in source
        assert 'actor_rollout_ref.model.lora_rank="$LORA_RANK"' in source
        assert 'actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"' in source
        assert '"${MODEL_LORA_ARGS[@]}"' in source


def test_paired_transfer_submitter_is_matched_and_complete():
    source = _read("scripts/submit_eval_qwen36_27b_sft_transfer.sh")
    assert "for variant in base sft" in source
    assert "BENCHMARKS=${BENCHMARKS:-gaia,bc,discovery}" in source
    assert "GAIA_METHODS=ctxgraph" in source
    assert "BC_METHODS=contextgraph" in source
    assert "DISCOVERYBENCH_METHODS=ctxgraph" in source
    assert "GAIA_CONTROLLER_ACTION_POLICY=balanced" in source
    assert "BC_CONTROLLER_ACTION_POLICY=balanced" in source
    assert "DISCOVERYBENCH_CONTROLLER_ACTION_POLICY=balanced" in source
    assert "LORA_RANK=${LORA_RANK:-32}" in source
    assert "LORA_ALPHA=${LORA_ALPHA:-64}" in source
    assert "EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:--1}" in source
    assert "DEFAULT_EVAL_TIME=00:30:00" in source
    assert "DEFAULT_EVAL_TIME=01:30:00" in source
    assert "GAIA_EVAL_TIME=${GAIA_EVAL_TIME:-$DEFAULT_EVAL_TIME}" in source
    assert "BC_EVAL_TIME=${BC_EVAL_TIME:-$DEFAULT_EVAL_TIME}" in source
    assert (
        "DISCOVERYBENCH_TIME_LIMIT="
        "${DISCOVERYBENCH_TIME_LIMIT:-$DEFAULT_EVAL_TIME}"
    ) in source
    assert 'GAIA_EVAL_TIME="$GAIA_EVAL_TIME"' in source
    assert 'BC_EVAL_TIME="$BC_EVAL_TIME"' in source
    assert 'DISCOVERYBENCH_TIME_LIMIT="$DISCOVERYBENCH_TIME_LIMIT"' in source
    assert "DRY_RUN=${DRY_RUN:-0}" in source
