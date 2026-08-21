#!/usr/bin/env python3
"""Validate structured graph traces in raw evaluator result artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph_trace import validate_graph_trace


def load_results(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("results")
    if not isinstance(payload, list):
        raise ValueError("expected a result list or an object with a results list")
    return [row for row in payload if isinstance(row, dict)]


def validate_results(results: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    counters = {
        "results": len(results),
        "runner_successes": 0,
        "valid_traces": 0,
        "trace_events": 0,
        "model_events": 0,
        "explicit_valid_model_ops": 0,
    }
    errors: list[str] = []
    for index, result in enumerate(results):
        if result.get("status") != "success":
            continue
        counters["runner_successes"] += 1
        trace = result.get("graph_trace")
        valid, trace_errors, metrics = validate_graph_trace(trace)
        task_id = result.get("task_id", index)
        if not valid:
            errors.append(f"{task_id}: " + "; ".join(trace_errors[:5]))
            continue
        counters["valid_traces"] += 1
        counters["trace_events"] += len(trace.get("events", []))
        counters["model_events"] += int(metrics.get("model_op_attempts", 0))
        counters["explicit_valid_model_ops"] += int(metrics.get("explicit_valid_model_ops", 0))
        expected_explicit = int((result.get("env_stats") or {}).get("graph_explicit_ops", 0) or 0)
        if expected_explicit != int(metrics.get("explicit_valid_model_ops", 0)):
            errors.append(
                f"{task_id}: graph_explicit_ops={expected_explicit} does not match "
                f"trace={metrics.get('explicit_valid_model_ops', 0)}"
            )
    if not counters["runner_successes"]:
        errors.append("no runner-success trajectory was available for trace validation")
    return counters, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    counters, errors = validate_results(load_results(args.input))
    print(json.dumps({**counters, "errors": errors}, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
