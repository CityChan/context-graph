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
    assert "PROMPT_LENGTH=8192" in source
    assert "RESPONSE_LENGTH=57344" in source
    assert "CONTEXT_LENGTH=65536" in source
    assert "TRAIN_BATCH_SIZE=32" in source
    assert "PPO_MINI_BATCH_SIZE=32" in source
    assert "ROLLOUT_N=8" in source
    assert "TRAIN_LR=1e-6" in source
    assert "USE_KL_LOSS=False" in source
    assert "CLIP_RATIO_HIGH=0.28" in source


def test_baseline_runner_exposes_matched_grpo_optimization_knobs():
    source = Path("scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh").read_text(encoding="utf-8")

    assert "USE_KL_LOSS=${USE_KL_LOSS:-True}" in source
    assert 'actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS"' in source
    assert 'actor_rollout_ref.actor.clip_ratio_low="$CLIP_RATIO_LOW"' in source
    assert 'actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH"' in source


def test_gaia_dev_split_is_deterministic_and_stratified():
    from scripts.split_gaia_validation_for_training import select_holdout_ids

    records = [(f"l1-{i}", "1") for i in range(10)] + [(f"l2-{i}", "2") for i in range(5)]
    first = select_holdout_ids(records, fraction=0.2, seed=42)
    second = select_holdout_ids(list(reversed(records)), fraction=0.2, seed=42)

    assert first == second
    assert len([task_id for task_id in first if task_id.startswith("l1-")]) == 2
    assert len([task_id for task_id in first if task_id.startswith("l2-")]) == 1
