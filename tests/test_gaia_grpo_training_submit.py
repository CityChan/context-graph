from pathlib import Path


def test_gaia_grpo_suite_uses_train_and_validation_splits():
    source = Path("scripts/submit_train_gaia_grpo_8b_5node_50step_64k.sh").read_text(encoding="utf-8")

    for suffix in ("", "_branch", "_graph"):
        assert f"data/gaia_train{suffix}.parquet" in source
        assert f"data/gaia_validation{suffix}.parquet" in source
    assert "for data_file in" in source
    assert "OPENAI_API_KEY" in source
    assert "GAIA_BUILD_TRAIN_IF_MISSING" in source
    assert "scripts/make_gaia_data.py --split train --out-dir data" in source


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
