#!/usr/bin/env python3
"""Audit persisted BrowseComp judge decisions from VeRL rollout JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def input_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.jsonl")))
            files.extend(sorted(path.rglob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return list(dict.fromkeys(files))


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("results") if "results" in payload else [payload]
    if not isinstance(payload, list):
        raise ValueError(f"{path} does not contain a JSON result list or JSONL records")
    return [record for record in payload if isinstance(record, dict)]


def binary_outcome(record: dict[str, Any]) -> float | None:
    """Return the per-episode outcome without mistaking batch metrics for labels."""
    env_stats = record.get("env_stats") or {}
    candidates = (
        env_stats.get("task_reward") if isinstance(env_stats, dict) else None,
        record.get("score"),
        record.get("task_reward"),
    )
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value in (0.0, 1.0):
            return value
    return None


def audit_results(paths: Iterable[Path], max_samples: int = 20) -> dict[str, Any]:
    files = input_files(paths)
    decisions: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    reward_mismatches: list[dict[str, Any]] = []
    record_count = 0

    for path in files:
        for row_index, record in enumerate(load_records(path)):
            record_count += 1
            audits = record.get("judge_audit") or []
            if not isinstance(audits, list):
                continue
            gen_uid = str(record.get("gen_uid") or f"{path}:{row_index}")
            for audit_index, audit in enumerate(audits):
                if not isinstance(audit, dict):
                    continue
                key = (gen_uid, audit_index)
                if key in seen:
                    continue
                seen.add(key)
                decisions.append({
                    "source": str(path),
                    "task_id": str(record.get("task_id", "unknown")),
                    "gen_uid": gen_uid,
                    **audit,
                })

            if audits:
                audit_scores = [float(audit.get("score", 0.0)) for audit in audits if isinstance(audit, dict)]
                if audit_scores:
                    expected = sum(audit_scores) / len(audit_scores)
                    observed = binary_outcome(record)
                    if observed is None or abs(observed - expected) > 1e-8:
                        reward_mismatches.append({
                            "source": str(path),
                            "task_id": str(record.get("task_id", "unknown")),
                            "gen_uid": gen_uid,
                            "expected": expected,
                            "observed": observed,
                        })

    methods = Counter(str(item.get("judge_method")) for item in decisions)
    positives = [item for item in decisions if float(item.get("score", 0.0)) > 0]
    non_strict_positives = [item for item in positives if not bool(item.get("strict_em"))]
    relaxed_only = [
        item for item in decisions
        if bool(item.get("relaxed_em")) and not bool(item.get("strict_em"))
    ]
    parse_failures = [item for item in decisions if item.get("judge_method") == "llm_parse_failure"]
    summary = {
        "files": len(files),
        "records": record_count,
        "unique_judge_decisions": len(decisions),
        "positive_decisions": len(positives),
        "strict_em_positives": sum(bool(item.get("strict_em")) for item in positives),
        "non_strict_positive_decisions_for_manual_review": len(non_strict_positives),
        "relaxed_only_diagnostic_matches": len(relaxed_only),
        "llm_parse_failures": len(parse_failures),
        "task_reward_audit_mismatches": len(reward_mismatches),
        "judge_methods": dict(sorted(methods.items())),
    }
    return {
        "summary": summary,
        "non_strict_positive_samples": non_strict_positives[:max_samples],
        "parse_failure_samples": parse_failures[:max_samples],
        "reward_mismatch_samples": reward_mismatches[:max_samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-integrity-error", action="store_true")
    args = parser.parse_args()
    report = audit_results(args.inputs, max_samples=max(0, args.max_samples))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    summary = report["summary"]
    if args.fail_on_integrity_error and (
        summary["llm_parse_failures"] or summary["task_reward_audit_mismatches"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
