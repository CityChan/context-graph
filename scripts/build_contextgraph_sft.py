#!/usr/bin/env python3
"""Build executor-verified ContextGraph multi-turn SFT parquet files.

The primary input is the JSON artifact written by ``scripts/eval_gaia.py
--save-messages``.  Only completed, task-correct trajectories with valid graph
operations are retained by default.  The output schema is consumed directly by
``verl.utils.dataset.multiturn_sft_dataset.MultiTurnSFTDataset``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


STRUCTURAL_GRAPH_FUNCTIONS = ("merge", "add_edge", "prune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="GAIA result JSON files")
    parser.add_argument("--output", required=True, help="Output training parquet")
    parser.add_argument("--validation-output", help="Optional validation parquet")
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-task-reward", type=float, default=1.0)
    parser.add_argument("--min-valid-graph-ops", type=int, default=1)
    parser.add_argument(
        "--min-structural-graph-ops",
        type=int,
        default=0,
        help="Require merge/add_edge/prune calls in the accepted conversation.",
    )
    parser.add_argument("--max-invalid-graph-ops", type=int, default=0)
    parser.add_argument("--teacher-provider", default="local_vllm")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Keep correct trajectories that did not explicitly finish.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Set enable_thinking=true in the SFT rows.",
    )
    return parser.parse_args()


def load_results(paths: Iterable[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            rows = payload["results"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"{path}: expected a result list or an object with a 'results' list")
        results.extend(row for row in rows if isinstance(row, dict))
    return results


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            return []
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            return []
        normalized.append({"role": role, "content": content})
    return normalized


def count_structural_graph_ops(messages: Iterable[dict[str, str]]) -> int:
    markers = tuple(f"<function={name}>" for name in STRUCTURAL_GRAPH_FUNCTIONS)
    return sum(
        message["content"].count(marker)
        for message in messages
        if message["role"] == "assistant"
        for marker in markers
    )


def result_to_sft_row(
    result: dict[str, Any],
    *,
    min_task_reward: float,
    min_valid_graph_ops: int,
    max_invalid_graph_ops: int,
    require_finish: bool,
    enable_thinking: bool,
    min_structural_graph_ops: int = 0,
    teacher_provider: str = "local_vllm",
    teacher_model: str | None = None,
) -> dict[str, Any] | None:
    if result.get("status") != "success":
        return None
    if float(result.get("task_reward", result.get("score", 0.0)) or 0.0) < min_task_reward:
        return None
    if require_finish and not bool(result.get("is_finish", False)):
        return None

    stats = result.get("env_stats") or {}
    valid_ops = int(stats.get("graph_explicit_ops", 0) or 0)
    invalid_ops = int(stats.get("graph_invalid_ops", 0) or 0)
    if valid_ops < min_valid_graph_ops or invalid_ops > max_invalid_graph_ops:
        return None
    if bool(stats.get("overlong", False)) or bool(stats.get("hit_token_limit", False)):
        return None

    messages = normalize_messages(result.get("messages"))
    if not messages or not any(message["role"] == "assistant" for message in messages):
        return None
    structural_ops = count_structural_graph_ops(messages)
    if structural_ops < min_structural_graph_ops:
        return None

    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    trajectory_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    resolved_teacher_model = teacher_model or str(result.get("teacher_model", "unknown"))
    return {
        "messages": messages,
        # ContextGraph currently uses textual XML function calls injected into
        # the system prompt, rather than tokenizer-native tool-call messages.
        "tools": [],
        "enable_thinking": bool(enable_thinking),
        "trajectory_id": trajectory_id,
        "task_id": str(result.get("task_id", "unknown")),
        "source": f"{teacher_provider}_contextgraph_teacher",
        "teacher_provider": teacher_provider,
        "teacher_model": resolved_teacher_model,
        "task_reward": float(result.get("task_reward", result.get("score", 0.0)) or 0.0),
        "graph_valid_ops": valid_ops,
        "graph_structural_ops": structural_ops,
        "graph_invalid_ops": invalid_ops,
        "graph_nodes": int(stats.get("graph_n_nodes", 0) or 0),
        "graph_edges": int(stats.get("graph_n_edges", 0) or 0),
    }


def build_rows(results: Iterable[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    counters = {"input": 0, "accepted": 0, "duplicates": 0, "rejected": 0}
    for result in results:
        counters["input"] += 1
        row = result_to_sft_row(
            result,
            min_task_reward=args.min_task_reward,
            min_valid_graph_ops=args.min_valid_graph_ops,
            max_invalid_graph_ops=args.max_invalid_graph_ops,
            require_finish=not args.allow_unfinished,
            enable_thinking=args.enable_thinking,
            min_structural_graph_ops=getattr(args, "min_structural_graph_ops", 0),
            teacher_provider=getattr(args, "teacher_provider", "local_vllm"),
            teacher_model=getattr(args, "teacher_model", None),
        )
        if row is None:
            counters["rejected"] += 1
            continue
        trajectory_id = row["trajectory_id"]
        if trajectory_id in rows_by_id:
            counters["duplicates"] += 1
            continue
        rows_by_id[trajectory_id] = row
        counters["accepted"] += 1
    return list(rows_by_id.values()), counters


def write_parquet(rows: list[dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in [0, 1)")

    rows, counters = build_rows(load_results(args.inputs), args)
    if not rows:
        raise SystemExit(f"No trajectories passed the filters: {json.dumps(counters, sort_keys=True)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    validation_count = 0
    if args.validation_output and len(rows) > 1 and args.validation_fraction > 0:
        validation_count = max(1, round(len(rows) * args.validation_fraction))
        validation_count = min(validation_count, len(rows) - 1)

    validation_rows = rows[:validation_count]
    training_rows = rows[validation_count:]
    write_parquet(training_rows, args.output)
    if args.validation_output and validation_rows:
        write_parquet(validation_rows, args.validation_output)

    print(json.dumps({
        **counters,
        "train_rows": len(training_rows),
        "validation_rows": len(validation_rows),
        "output": str(Path(args.output)),
        "validation_output": str(Path(args.validation_output)) if validation_rows else None,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
