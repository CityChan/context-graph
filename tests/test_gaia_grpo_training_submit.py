from pathlib import Path


def test_gaia_grpo_suite_uses_matched_train_and_holdout_splits():
    source = Path("scripts/submit_train_gaia_grpo_8b_5node_50step_64k.sh").read_text(encoding="utf-8")

    for suffix in ("", "_branch", "_graph"):
        assert f"data/gaia_train{suffix}.parquet" in source
        assert f"data/gaia_holdout{suffix}.parquet" in source
    assert "for data_file in" in source
    assert "OPENAI_API_KEY" in source
    assert "GAIA_PREPARE_DEV_SPLIT_IF_MISSING" in source
    assert "scripts/split_gaia_validation_for_training.py" in source


def test_gaia_grpo_suite_is_matched_64k_training():
    source = Path("scripts/submit_train_gaia_grpo_8b_5node_50step_64k.sh").read_text(encoding="utf-8")

    assert "GAIA_TRAIN_METHODS=${GAIA_TRAIN_METHODS:-baseline,foldagent,ctxgraph}" in source
    assert "TRAINER_VAL_ONLY=False" in source
    assert "TOTAL_TRAINING_STEPS=$GAIA_TRAIN_STEPS" in source
    assert "GAIA_CONTEXT_TAG=${GAIA_CONTEXT_TAG:-64k}" in source
    assert "GAIA_PROMPT_LENGTH=${GAIA_PROMPT_LENGTH:-8192}" in source
    assert "GAIA_RESPONSE_LENGTH=${GAIA_RESPONSE_LENGTH:-57344}" in source
    assert "GAIA_CONTEXT_LENGTH=${GAIA_CONTEXT_LENGTH:-65536}" in source
    assert "PROMPT_LENGTH=$GAIA_PROMPT_LENGTH" in source
    assert "RESPONSE_LENGTH=$GAIA_RESPONSE_LENGTH" in source
    assert "CONTEXT_LENGTH=$GAIA_CONTEXT_LENGTH" in source
    assert "TRAIN_BATCH_SIZE=32" in source
    assert "PPO_MINI_BATCH_SIZE=32" in source
    assert "ROLLOUT_N=8" in source
    assert "ENTROPY_FROM_LOGITS_WITH_CHUNKING=True" in source
    assert "TRAIN_LR=1e-6" in source
    assert "USE_KL_LOSS=False" in source
    assert "CLIP_RATIO_HIGH=0.28" in source


def test_gaia_grpo_32k_suite_reuses_matched_submitter_with_32k_budget():
    source = Path("scripts/submit_train_gaia_grpo_8b_5node_50step_32k.sh").read_text(encoding="utf-8")
    assert "GAIA_CONTEXT_TAG=32k" in source
    assert "GAIA_PROMPT_LENGTH=8192" in source
    assert "GAIA_RESPONSE_LENGTH=24576" in source
    assert "GAIA_CONTEXT_LENGTH=32768" in source
    assert "submit_train_gaia_grpo_8b_5node_50step_64k.sh" in source


def test_baseline_runner_exposes_matched_grpo_optimization_knobs():
    source = Path("scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh").read_text(encoding="utf-8")

    assert "USE_KL_LOSS=${USE_KL_LOSS:-True}" in source
    assert 'actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS"' in source
    assert 'actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW"' in source
    assert 'actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH"' in source


def test_gaia_training_runners_bound_entropy_memory_and_bypass_local_proxy():
    runners = (
        "scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh",
        "scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
    )
    for runner in runners:
        source = Path(runner).read_text(encoding="utf-8")
        assert "ENTROPY_FROM_LOGITS_WITH_CHUNKING=${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-True}" in source
        assert 'actor_rollout_ref.actor.entropy_from_logits_with_chunking="$ENTROPY_FROM_LOGITS_WITH_CHUNKING"' in source
        assert "export NO_PROXY=" in source
        assert "export no_proxy=\"$NO_PROXY\"" in source
        assert "curl --noproxy '*' -fsS" in source

    baseline = Path(runners[0]).read_text(encoding="utf-8")
    assert "BC_SEARCH_TIMEOUT_SECONDS=${BC_SEARCH_TIMEOUT_SECONDS:-600}" in baseline
    assert 'probe "waiting for search server /search probe' in baseline
    assert 'if [ "$SEARCH_OK" != "1" ]; then' in baseline


def test_gaia_dev_split_is_deterministic_and_stratified():
    from scripts.split_gaia_validation_for_training import select_holdout_ids

    records = [(f"l1-{i}", "1") for i in range(10)] + [(f"l2-{i}", "2") for i in range(5)]
    first = select_holdout_ids(records, fraction=0.2, seed=42)
    second = select_holdout_ids(list(reversed(records)), fraction=0.2, seed=42)

    assert first == second
    assert len([task_id for task_id in first if task_id.startswith("l1-")]) == 2
    assert len([task_id for task_id in first if task_id.startswith("l2-")]) == 1
