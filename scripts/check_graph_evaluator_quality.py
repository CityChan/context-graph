#!/usr/bin/env python3
"""Gate a calibrated graph evaluator against a constant-prevalence baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def quality_result(
    payload: dict, *, min_auroc: float, max_brier_ratio: float
) -> dict[str, float | int | bool]:
    calibrated = payload["calibrated"]
    baseline = payload["constant_prevalence_baseline"]
    auroc = float(calibrated["auroc"])
    brier = float(calibrated["brier"])
    baseline_brier = float(baseline["brier"])
    brier_ratio = brier / baseline_brier if baseline_brier > 0.0 else math.inf
    return {
        "validation_rows": int(payload["validation_rows"]),
        "validation_questions": int(payload["validation_questions"]),
        "calibrated_auroc": auroc,
        "calibrated_brier": brier,
        "baseline_brier": baseline_brier,
        "brier_ratio": brier_ratio,
        "min_auroc": min_auroc,
        "max_brier_ratio": max_brier_ratio,
        "passed": auroc >= min_auroc and brier_ratio <= max_brier_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--min-auroc", type=float, default=0.55)
    parser.add_argument("--max-brier-ratio", type=float, default=1.0)
    args = parser.parse_args()

    payload = json.loads(args.evaluation.read_text(encoding="utf-8"))
    result = quality_result(
        payload,
        min_auroc=args.min_auroc,
        max_brier_ratio=args.max_brier_ratio,
    )
    result["evaluation"] = str(args.evaluation)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
