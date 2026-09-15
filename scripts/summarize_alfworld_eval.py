#!/usr/bin/env python3
"""Extract the final ALFWorld validation metrics from a VERL console log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


METRICS = (
    "reward",
    "task_reward",
    "finish_rate",
    "no_finish_rate",
    "token_limit_rate",
    "max_turn_rate",
    "avg_num_turns",
    "main_turn",
    "graph_total_ops",
    "graph_explicit_ops",
    "graph_n_pruned",
    "structured_graph_controller",
    "controller_structural_policy",
)


def extract_metrics(text: str) -> dict[str, float]:
    step_lines = [line for line in text.splitlines() if "step:0 -" in line and "val/task_reward:" in line]
    if not step_lines:
        raise ValueError("no completed step:0 validation metrics line found")
    line = step_lines[-1]
    result: dict[str, float] = {}
    for metric in METRICS:
        pattern = rf"(?:^| - )val/{re.escape(metric)}:(?:np\.float64\()?([-+0-9.eE]+)"
        match = re.search(pattern, line)
        if match:
            result[metric] = float(match.group(1))
    if "task_reward" not in result:
        raise ValueError("completed validation line has no val/task_reward")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    metrics = extract_metrics(args.log.read_text(encoding="utf-8", errors="replace"))
    summary = {
        "samples": args.samples,
        "successes": round(metrics["task_reward"] * args.samples),
        "success_rate": metrics["task_reward"],
        "metrics": metrics,
        "log": str(args.log),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
