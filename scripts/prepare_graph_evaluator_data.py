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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("results")
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
                key = (row["task_id"], row["view_hash"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def validation_task(task_id: str, validation_fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < validation_fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")

    rows = build_rows(args.inputs)
    if not rows:
        raise ValueError("no graph evaluator rows were extracted")
    train = [
        row for row in rows
        if not validation_task(row["question_hash"], args.validation_fraction, args.seed)
    ]
    validation = [
        row for row in rows
        if validation_task(row["question_hash"], args.validation_fraction, args.seed)
    ]
    if not train or not validation:
        raise ValueError("question-disjoint split produced an empty partition; add more tasks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "graph_evaluator_train.parquet"
    validation_path = args.output_dir / "graph_evaluator_validation.parquet"
    pd.DataFrame(train).to_parquet(train_path, index=False)
    pd.DataFrame(validation).to_parquet(validation_path, index=False)
    manifest = {
        "schema_version": "contextgraph.graph_evaluator_data.v1",
        "inputs": [str(path) for path in args.inputs],
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_questions": len({row["question_hash"] for row in train}),
        "validation_questions": len({row["question_hash"] for row in validation}),
        "question_overlap": len(
            {row["question_hash"] for row in train}
            & {row["question_hash"] for row in validation}
        ),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_output": str(train_path),
        "validation_output": str(validation_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
