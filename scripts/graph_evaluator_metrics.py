"""Dependency-light binary metrics for graph evaluator validation."""

from __future__ import annotations

import numpy as np


def positive_probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-4)
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC requires both binary outcome classes")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def grouped_binary_ranking_metrics(
    labels: np.ndarray, scores: np.ndarray, group_ids: np.ndarray
) -> dict[str, float | int]:
    """Measure positive/negative ranking only within mixed-label groups.

    The macro AUROC gives every question equal weight. The pair-weighted AUROC
    weights questions by their number of positive/negative comparisons and is
    reported as a diagnostic, not used as the primary quality gate.
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    group_ids = np.asarray(group_ids).astype(str)
    if not (len(labels) == len(scores) == len(group_ids)):
        raise ValueError("labels, scores, and group IDs must have equal length")

    group_aurocs: list[float] = []
    pair_counts: list[int] = []
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        group_labels = labels[mask]
        positive_count = int((group_labels == 1).sum())
        negative_count = int((group_labels == 0).sum())
        if positive_count == 0 or negative_count == 0:
            continue
        group_aurocs.append(binary_auroc(group_labels, scores[mask]))
        pair_counts.append(positive_count * negative_count)

    if not group_aurocs:
        return {
            "mixed_groups": 0,
            "positive_negative_pairs": 0,
            "macro_auroc": 0.0,
            "pair_weighted_auroc": 0.0,
        }
    return {
        "mixed_groups": len(group_aurocs),
        "positive_negative_pairs": int(sum(pair_counts)),
        "macro_auroc": float(np.mean(group_aurocs)),
        "pair_weighted_auroc": float(
            np.average(group_aurocs, weights=np.asarray(pair_counts, dtype=np.float64))
        ),
    }


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    predictions = (probabilities >= 0.5).astype(np.int64)
    positive_accuracy = float((predictions[labels == 1] == 1).mean())
    negative_accuracy = float((predictions[labels == 0] == 0).mean())
    expected_calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        in_bin = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if in_bin.any():
            expected_calibration_error += float(in_bin.mean()) * abs(
                float(probabilities[in_bin].mean()) - float(labels[in_bin].mean())
            )
    return {
        "nll": float(
            -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)).mean()
        ),
        "brier": float(np.square(probabilities - labels).mean()),
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": (positive_accuracy + negative_accuracy) / 2.0,
        "positive_recall": positive_accuracy,
        "negative_recall": negative_accuracy,
        "auroc": binary_auroc(labels, probabilities),
        "ece_10_bin": expected_calibration_error,
    }
