#!/usr/bin/env python3
"""Summarize Search-R1 validation scores and basic GRPO update health from logs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


SEARCH_BENCHMARKS = (
    "searchR1_nq",
    "searchR1_triviaqa",
    "searchR1_popqa",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
    "searchR1_bamboogle",
)

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def extract_numbers(text: str, key: str) -> list[float]:
    pattern = re.compile(
        rf"(?:['\"])?{re.escape(key)}(?:['\"])?\s*[:=]\s*(?:np\.float\d*\()?({NUMBER})"
    )
    return [float(value) for value in pattern.findall(text)]


def extract_last_number(text: str, key: str) -> float | None:
    values = extract_numbers(text, key)
    return values[-1] if values else None


def summarize_log(
    path: Path,
    sources: tuple[str, ...] = SEARCH_BENCHMARKS,
) -> tuple[dict[str, float], dict[str, float | None], dict[str, list[float]]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    scores = {}
    score_history = {}
    for source in sources:
        key = f"val/{source}/test_score"
        values = extract_numbers(text, key)
        if values:
            scores[source] = values[-1]
            score_history[source] = values

    health_keys = (
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/kl_loss",
        "critic/score/mean",
        "reward/mean",
    )
    health = {key: extract_last_number(text, key) for key in health_keys}
    return scores, health, score_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--require-training-health",
        action="store_true",
        help="Require finite PG loss, gradient norm, KL loss, and reward metrics.",
    )
    parser.add_argument(
        "--require-all-benchmarks",
        action="store_true",
        help="Require all seven Search-R1 validation metrics.",
    )
    parser.add_argument(
        "--sources",
        default=",".join(SEARCH_BENCHMARKS),
        help="Comma-separated validation data_source values to audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = tuple(source.strip() for source in args.sources.split(",") if source.strip())
    if not sources:
        raise SystemExit("--sources must contain at least one source")
    failed = False
    for path in args.logs:
        scores, health, score_history = summarize_log(path, sources)
        print(f"[{path}]")
        for source in sources:
            label = source.removeprefix("searchR1_")
            value = scores.get(source)
            print(f"  {label:20s} {value:.4f}" if value is not None else f"  {label:20s} MISSING")
        if scores:
            print(f"  {'macro_average':20s} {sum(scores.values()) / len(scores):.4f}")
        paired_history = [values for values in score_history.values() if len(values) >= 2]
        if len(paired_history) == len(sources):
            initial = sum(values[0] for values in paired_history) / len(paired_history)
            final = sum(values[-1] for values in paired_history) / len(paired_history)
            print(f"  {'macro_change':20s} {initial:.4f} -> {final:.4f} ({final - initial:+.4f})")
        for key, value in health.items():
            if value is not None:
                print(f"  {key:20s} {value:.8g}")
                failed |= not math.isfinite(value)
        if args.require_training_health:
            required = ("actor/pg_loss", "actor/grad_norm", "actor/kl_loss")
            missing_health = [key for key in required if health[key] is None]
            if health["critic/score/mean"] is None and health["reward/mean"] is None:
                missing_health.append("critic/score/mean or reward/mean")
            if missing_health:
                failed = True
                print(f"  ERROR: missing training health metrics: {', '.join(missing_health)}")
            text = path.read_text(encoding="utf-8", errors="replace")
            grad_norms = extract_numbers(text, "actor/grad_norm")
            if grad_norms and not any(value > 0 for value in grad_norms):
                failed = True
                print("  ERROR: actor/grad_norm was never positive")
            for key in required:
                nonfinite = re.search(
                    rf"(?:['\"])?{re.escape(key)}(?:['\"])?\s*[:=]\s*(?:np\.float\d*\()?[-+]?(?:nan|inf)",
                    text,
                    flags=re.IGNORECASE,
                )
                if nonfinite:
                    failed = True
                    print(f"  ERROR: non-finite value logged for {key}")
        if args.require_all_benchmarks and len(scores) != len(sources):
            failed = True
            print(f"  ERROR: found {len(scores)}/{len(sources)} benchmark scores")
        elif len(scores) not in (0, len(sources)):
            failed = True
            print(f"  ERROR: found {len(scores)}/{len(sources)} benchmark scores")
        print()
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
