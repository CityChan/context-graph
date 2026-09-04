from pathlib import Path

from scripts.prepare_graph_evaluator_data import (
    binary_class_counts,
    has_both_classes,
    sample_question_groups,
    split_rows,
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
    assert "class ClassWeightedTrainer" in source
    assert "balanced_class_weights(train_frame.label.to_numpy())" in source
    assert "cross_entropy" in source


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
