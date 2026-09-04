#!/usr/bin/env python3
"""Build question-disjoint BCE data for the frozen GraphRPO evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_results(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # VeRL trainer.rollout_data_dir writes one result object per line.
        # Accept that native JSONL format as well as evaluation JSON arrays.
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("results") if "results" in payload else [payload]
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a result list or an object with results")
    return [row for row in payload if isinstance(row, dict)]


def _question_from_trace(trace: dict[str, Any]) -> str:
    initial = trace.get("initial_graph") or {}
    root_id = initial.get("root_id")
    for node in initial.get("nodes", []):
        if node.get("id") == root_id:
            return str(node.get("content", "")).strip()
    return ""


def result_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    if result.get("status") not in (None, "success"):
        return []
    trace = result.get("graph_trace")
    if not isinstance(trace, dict):
        return []
    label = result.get("task_reward", (result.get("env_stats") or {}).get("task_reward"))
    try:
        label = float(label)
    except (TypeError, ValueError):
        return []
    if label not in (0.0, 1.0):
        return []

    question = str(result.get("question") or _question_from_trace(trace)).strip()
    if not question:
        return []
    task_id = str(result.get("task_id") or hashlib.sha256(question.encode()).hexdigest())
    episode_id = str(result.get("gen_uid") or result.get("uid") or "").strip()
    if not episode_id:
        episode_fingerprint = json.dumps(result, ensure_ascii=False, sort_keys=True)
        episode_id = hashlib.sha256(episode_fingerprint.encode("utf-8")).hexdigest()
    question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
    views: list[str] = []
    for event in trace.get("events", []):
        if not isinstance(event, dict):
            continue
        for key in ("rendered_before", "rendered_after"):
            view = event.get(key)
            if isinstance(view, str) and view.strip():
                views.append(view)
    final_view = result.get("graph_state")
    if isinstance(final_view, str) and final_view.strip():
        views.append(final_view)

    unique: dict[str, dict[str, Any]] = {}
    for view in views:
        view_hash = hashlib.sha256(view.encode("utf-8")).hexdigest()
        unique[view_hash] = {
            "task_id": task_id,
            "episode_id": episode_id,
            "question_hash": question_hash,
            "question": question,
            "graph_view": view,
            "label": int(label),
            "view_hash": view_hash,
        }
    return list(unique.values())


def build_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for result in load_results(path):
            for row in result_rows(result):
                # Main/branch streams may duplicate an episode, while two
                # independently sampled episodes can legitimately attach
                # different outcomes to the same question/view.
                key = (row["episode_id"], row["view_hash"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def validation_task(task_id: str, validation_fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < validation_fraction


def split_rows(
    rows: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [
        row
        for row in rows
        if not validation_task(row["question_hash"], validation_fraction, seed)
    ]
    validation = [
        row
        for row in rows
        if validation_task(row["question_hash"], validation_fraction, seed)
    ]
    return train, validation


def binary_class_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "negative": sum(int(row["label"]) == 0 for row in rows),
        "positive": sum(int(row["label"]) == 1 for row in rows),
    }


def has_both_classes(rows: list[dict[str, Any]]) -> bool:
    return {int(row["label"]) for row in rows} == {0, 1}


def sample_question_groups(
    rows: list[dict[str, Any]], max_questions: int, seed: int
) -> list[dict[str, Any]]:
    question_hashes = {str(row["question_hash"]) for row in rows}
    if max_questions <= 0 or len(question_hashes) <= max_questions:
        return rows
    ordered = sorted(
        question_hashes,
        key=lambda value: hashlib.sha256(f"sample:{seed}:{value}".encode()).digest(),
    )
    selected = set(ordered[:max_questions])
    return [row for row in rows if str(row["question_hash"]) in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auto-seed-attempts",
        type=int,
        default=1,
        help="Try consecutive split seeds until all requested constraints hold.",
    )
    parser.add_argument(
        "--require-both-classes",
        action="store_true",
        help="Require both binary labels in both question-disjoint partitions.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=0,
        help="Deterministically cap complete question groups; 0 keeps all questions.",
    )
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")
    if args.auto_seed_attempts < 1:
        raise ValueError("auto seed attempts must be positive")

    all_rows = build_rows(args.inputs)
    if not all_rows:
        raise ValueError("no graph evaluator rows were extracted")
    rows = sample_question_groups(all_rows, args.max_questions, args.seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    selected_seed: int | None = None
    for candidate_seed in range(args.seed, args.seed + args.auto_seed_attempts):
        candidate_train, candidate_validation = split_rows(
            rows, args.validation_fraction, candidate_seed
        )
        if not candidate_train or not candidate_validation:
            continue
        if args.require_both_classes and not (
            has_both_classes(candidate_train) and has_both_classes(candidate_validation)
        ):
            continue
        train, validation = candidate_train, candidate_validation
        selected_seed = candidate_seed
        break
    if selected_seed is None:
        question_count = len({row["question_hash"] for row in rows})
        positive_questions = len(
            {row["question_hash"] for row in rows if int(row["label"]) == 1}
        )
        negative_questions = len(
            {row["question_hash"] for row in rows if int(row["label"]) == 0}
        )
        raise ValueError(
            "could not produce a valid question-disjoint split after "
            f"{args.auto_seed_attempts} seed attempts; rows={len(rows)}, "
            f"questions={question_count}, positive_questions={positive_questions}, "
            f"negative_questions={negative_questions}. Collect more independently "
            "sampled tasks, especially successful tasks."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "graph_evaluator_train.parquet"
    validation_path = args.output_dir / "graph_evaluator_validation.parquet"
    pd.DataFrame(train).to_parquet(train_path, index=False)
    pd.DataFrame(validation).to_parquet(validation_path, index=False)
    manifest = {
        "schema_version": "contextgraph.graph_evaluator_data.v1",
        "inputs": [str(path) for path in args.inputs],
        "input_rows": len(all_rows),
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_questions": len({row["question_hash"] for row in train}),
        "validation_questions": len({row["question_hash"] for row in validation}),
        "question_overlap": len(
            {row["question_hash"] for row in train}
            & {row["question_hash"] for row in validation}
        ),
        "requested_seed": args.seed,
        "seed": selected_seed,
        "validation_fraction": args.validation_fraction,
        "max_questions": args.max_questions,
        "require_both_classes": args.require_both_classes,
        "all_class_counts": binary_class_counts(rows),
        "train_class_counts": binary_class_counts(train),
        "validation_class_counts": binary_class_counts(validation),
        "train_output": str(train_path),
        "validation_output": str(validation_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
