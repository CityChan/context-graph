#!/usr/bin/env python3
"""Summarize per-query DiscoveryBench HMS audit records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(result_dir: str | Path) -> dict:
    directory = Path(result_dir)
    files = sorted(directory.glob("*.json"))
    scores = []
    judge_errors = 0
    format_only = 0
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        score = record.get("score")
        if isinstance(score, dict) and "final_score" in score:
            scores.append(float(score["final_score"]))
        elif str(record.get("detail", "")).startswith("HMS judge failed"):
            judge_errors += 1
        else:
            format_only += 1
    return {
        "result_dir": str(directory),
        "audit_records": len(files),
        "hms_scored": len(scores),
        "judge_errors": judge_errors,
        "format_only_or_unscored": format_only,
        "mean_hms": sum(scores) / len(scores) if scores else None,
        "min_hms": min(scores) if scores else None,
        "max_hms": max(scores) if scores else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    args = parser.parse_args()
    print(json.dumps(summarize(args.result_dir), indent=2))


if __name__ == "__main__":
    main()
