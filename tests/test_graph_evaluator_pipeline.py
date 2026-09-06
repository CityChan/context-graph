from pathlib import Path

import numpy as np

from scripts.prepare_graph_evaluator_data import (
    binary_class_counts,
    has_both_classes,
    mixed_question_count,
    sample_question_groups,
    split_meets_constraints,
    split_rows,
)
from scripts.graph_evaluator_metrics import (
    binary_auroc,
    grouped_binary_ranking_metrics,
    positive_probabilities,
    probability_metrics,
)
from scripts.check_graph_evaluator_quality import quality_result


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


def test_split_constraints_require_enough_mixed_questions():
    train = [
        {"question_hash": question, "label": label}
        for question in ("q1", "q2", "q3")
        for label in (0, 1)
    ]
    validation = [
        {"question_hash": question, "label": label}
        for question in ("q4", "q5")
        for label in (0, 1)
    ]
    assert mixed_question_count(train) == 3
    assert mixed_question_count(validation) == 2
    assert split_meets_constraints(
        train,
        validation,
        require_both_classes=True,
        min_train_mixed_questions=3,
        min_validation_mixed_questions=2,
    )
    assert not split_meets_constraints(
        train,
        validation,
        require_both_classes=True,
        min_train_mixed_questions=3,
        min_validation_mixed_questions=3,
    )


def test_training_uses_class_balanced_loss():
    source = Path("scripts/train_graph_evaluator.py").read_text(encoding="utf-8")
    assert source.index('os.environ.setdefault("USE_TF", "0")') < source.index(
        "from transformers import"
    )
    assert "class ClassWeightedTrainer" in source
    assert "balanced_class_weights(train_frame.label.to_numpy())" in source
    assert "cross_entropy" in source
    assert 'choices=("none", "wandb")' in source
    assert 'f"graph_evaluator/{section}_{metric}"' in source
    assert "within_question_episode_ranking" in source


def test_graph_evaluator_quality_gate_requires_discrimination_and_brier_gain():
    payload = {
        "validation_rows": 20,
        "validation_questions": 5,
        "calibrated": {"auroc": 0.7, "brier": 0.15},
        "constant_prevalence_baseline": {"brier": 0.2},
        "within_question_episode_ranking": {
            "mixed_groups": 5,
            "macro_auroc": 0.65,
        },
    }
    kwargs = {
        "min_auroc": 0.55,
        "max_brier_ratio": 1.0,
        "min_mixed_questions": 5,
        "min_within_question_auroc": 0.55,
    }
    assert quality_result(payload, **kwargs)["passed"]
    payload["calibrated"]["auroc"] = 0.5
    assert not quality_result(payload, **kwargs)["passed"]
    payload["calibrated"]["auroc"] = 0.7
    payload["within_question_episode_ranking"]["mixed_groups"] = 4
    assert not quality_result(payload, **kwargs)["passed"]


def test_graph_evaluator_metrics_cover_discrimination_and_calibration():
    labels = np.array([0, 0, 1, 1])
    logits = np.array([[4.0, 0.0], [3.0, 1.0], [1.0, 3.0], [0.0, 4.0]])
    probabilities = positive_probabilities(logits, temperature=2.0)
    metrics = probability_metrics(labels, probabilities)
    assert binary_auroc(labels, probabilities) == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert 0.0 <= metrics["ece_10_bin"] <= 1.0


def test_grouped_ranking_uses_only_within_question_comparisons():
    labels = np.array([0, 1, 0, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    questions = np.array(["q1", "q1", "q2", "q2", "q3"])
    metrics = grouped_binary_ranking_metrics(labels, scores, questions)
    assert metrics["mixed_groups"] == 2
    assert metrics["positive_negative_pairs"] == 2
    assert metrics["macro_auroc"] == 0.5
    assert metrics["pair_weighted_auroc"] == 0.5


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


def test_browsecomp_rollout_evaluator_pilot_is_wandb_audited_and_gated():
    source = Path(
        "scripts/pilot_train_graph_evaluator_from_bc_rollouts_4node_idev.sh"
    ).read_text(encoding="utf-8")
    assert "audit_bc_judge_results.py" in source
    assert "prepare_graph_evaluator_data.py" in source
    assert "--require-both-classes" in source
    assert "--min-validation-mixed-questions" in source
    assert "--report-to wandb" in source
    assert "check_graph_evaluator_quality.py" in source
    assert "--min-within-question-auroc" in source
    assert "bc_test.parquet" in source
    assert "does not update the actor" in source
