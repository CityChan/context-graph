from pathlib import Path

import numpy as np

from scripts.prepare_graph_evaluator_data import (
    binary_class_counts,
    has_both_classes,
    sample_question_groups,
    split_rows,
)
from scripts.graph_evaluator_metrics import (
    binary_auroc,
    positive_probabilities,
    probability_metrics,
)


def test_split_is_question_disjoint_and_can_keep_both_classes():
    rows = [
        {"question_hash": f"q{question_index}", "label": label}
        for question_index in range(12)
        for label in (0, 1)
    ]
    selected = None
    for seed in range(100):
        train, validation = split_rows(rows, 0.25, seed)
        if train and validation and has_both_classes(train) and has_both_classes(validation):
            selected = train, validation
            break
    assert selected is not None
    train, validation = selected
    assert {row["question_hash"] for row in train}.isdisjoint(
        {row["question_hash"] for row in validation}
    )
    assert binary_class_counts(train)["positive"] > 0
    assert binary_class_counts(validation)["negative"] > 0


def test_question_cap_keeps_complete_groups_deterministically():
    rows = [
        {"question_hash": f"q{question_index}", "label": label}
        for question_index in range(10)
        for label in (0, 1)
    ]
    sampled = sample_question_groups(rows, max_questions=4, seed=42)
    selected_questions = {row["question_hash"] for row in sampled}
    assert len(selected_questions) == 4
    assert len(sampled) == 8
    assert sampled == sample_question_groups(rows, max_questions=4, seed=42)


def test_training_uses_class_balanced_loss():
    source = Path("scripts/train_graph_evaluator.py").read_text(encoding="utf-8")
    assert source.index('os.environ.setdefault("USE_TF", "0")') < source.index(
        "from transformers import"
    )
    assert "class ClassWeightedTrainer" in source
    assert "balanced_class_weights(train_frame.label.to_numpy())" in source
    assert "cross_entropy" in source


def test_graph_evaluator_metrics_cover_discrimination_and_calibration():
    labels = np.array([0, 0, 1, 1])
    logits = np.array([[4.0, 0.0], [3.0, 1.0], [1.0, 3.0], [0.0, 4.0]])
    probabilities = positive_probabilities(logits, temperature=2.0)
    metrics = probability_metrics(labels, probabilities)
    assert binary_auroc(labels, probabilities) == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert 0.0 <= metrics["ece_10_bin"] <= 1.0


def test_raw_sft_pilot_trains_calibrates_and_probes():
    source = Path(
        "scripts/pilot_train_graph_rpo_evaluator_from_sft_raw.sh"
    ).read_text(encoding="utf-8")
    assert "interactive_results_*.json" in source
    assert "gaia_results_*.json" in source
    assert "--require-both-classes" in source
    assert "scripts/train_graph_evaluator.py" in source
    assert "graph_rpo_calibration.json" in source
    assert "scripts/serve_graph_evaluator.py" in source
    assert "context-graph-evaluator-data" in source
    assert "context-graph-evaluators" in source
    assert '"$RAW_SFT_REAL"|"$RAW_SFT_REAL"/*' in source
    assert "REUSE_LATEST_PREPARED_DATA" in source
    assert "export USE_TF=0" in source


def test_browsecomp_target_evaluator_build_is_train_only_and_policy_frozen():
    source = Path(
        "scripts/build_bc_graph_rpo_evaluator_qwen3_8b_4node.sh"
    ).read_text(encoding="utf-8")
    assert "data/bc_train.parquet" in source
    assert "evaluator collection must not use the BrowseComp test split" in source
    assert "TRAINER_VAL_ONLY=True" in source
    assert "VAL_DO_SAMPLE=True" in source
    assert "VALIDATION_DATA_DIR" in source
    assert "scripts/audit_bc_judge_results.py" in source
    assert "--require-both-classes" in source
    assert "GRAPH_EVALUATOR_PRETRAINED_MODEL" in source
    assert "graph_rpo_evaluation.json" in source
    assert 'if [ "$REUSE_TARGET_ROLLOUTS" = "1" ]' in source
    assert source.index("export JUDGE_MODEL=${JUDGE_MODEL:-gpt-5-nano}") < source.index(
        'if [ "$REUSE_TARGET_ROLLOUTS" != "1" ]'
    )
    trainer_source = Path("verl/trainer/ppo/ray_trainer.py").read_text(
        encoding="utf-8"
    )
    assert (
        'test_batch.meta_info["temperature"] = '
        "self.config.actor_rollout_ref.rollout.val_kwargs.temperature"
        in trainer_source
    )
