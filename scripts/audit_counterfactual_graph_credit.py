#!/usr/bin/env python3
"""Audit persisted paired-counterfactual GraphRPO probes and edit deltas."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


BACKEND = "old_policy_counterfactual_qa"


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


def answer_format_valid(response: Any) -> bool:
    matches = re.findall(
        r"<answer>(.*?)</answer>",
        str(response or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return len(matches) == 1 and bool(matches[0].strip())


def audit_results(paths: Iterable[Path], max_samples: int = 20) -> dict[str, Any]:
    files = input_files(paths)
    seen_episodes: set[str] = set()
    integrity_errors: list[dict[str, Any]] = []
    malformed_samples: list[dict[str, Any]] = []
    record_count = 0
    episodes_with_trace = 0
    episodes_with_credit = 0
    edit_count = 0
    probe_count = 0
    tagged_count = 0
    positive_count = 0
    paired_reward_differences = 0
    nonzero_edit_count = 0
    delta_sum = 0.0
    delta_abs_sum = 0.0

    for path in files:
        for row_index, record in enumerate(load_records(path)):
            record_count += 1
            gen_uid = str(record.get("gen_uid") or f"{path}:{row_index}")
            if gen_uid in seen_episodes:
                continue
            seen_episodes.add(gen_uid)
            trace = record.get("graph_trace")
            if not isinstance(trace, dict):
                continue
            episodes_with_trace += 1
            events = [
                event
                for event in trace.get("events", [])
                if isinstance(event, dict)
                and event.get("graph_rpo_credit_backend") == BACKEND
            ]
            if events:
                episodes_with_credit += 1

            for event in events:
                edit_count += 1
                seq = event.get("seq")
                before_responses = event.get(
                    "graph_rpo_counterfactual_before_responses", []
                )
                after_responses = event.get(
                    "graph_rpo_counterfactual_after_responses", []
                )
                before_rewards = event.get(
                    "graph_rpo_counterfactual_before_rewards", []
                )
                after_rewards = event.get(
                    "graph_rpo_counterfactual_after_rewards", []
                )
                arrays = (
                    before_responses,
                    after_responses,
                    before_rewards,
                    after_rewards,
                )
                if not all(isinstance(value, list) for value in arrays):
                    integrity_errors.append(
                        {"source": str(path), "gen_uid": gen_uid, "seq": seq,
                         "error": "probe payload is not list-valued"}
                    )
                    continue
                lengths = {len(value) for value in arrays}
                if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
                    integrity_errors.append(
                        {"source": str(path), "gen_uid": gen_uid, "seq": seq,
                         "error": "before/after probe lengths differ or are empty",
                         "lengths": [len(value) for value in arrays]}
                    )
                    continue

                for side, responses, rewards in (
                    ("before", before_responses, before_rewards),
                    ("after", after_responses, after_rewards),
                ):
                    for sample_index, (response, reward) in enumerate(
                        zip(responses, rewards)
                    ):
                        probe_count += 1
                        valid = answer_format_valid(response)
                        tagged_count += int(valid)
                        try:
                            numeric_reward = float(reward)
                        except (TypeError, ValueError):
                            numeric_reward = float("nan")
                        if numeric_reward not in (0.0, 1.0):
                            integrity_errors.append(
                                {"source": str(path), "gen_uid": gen_uid,
                                 "seq": seq, "side": side,
                                 "sample_index": sample_index,
                                 "error": f"non-binary reward: {reward!r}"}
                            )
                        positive_count += int(numeric_reward == 1.0)
                        if not valid and len(malformed_samples) < max_samples:
                            malformed_samples.append(
                                {"source": str(path), "gen_uid": gen_uid,
                                 "seq": seq, "side": side,
                                 "sample_index": sample_index,
                                 "response_tail": str(response or "")[-500:]}
                            )

                paired_reward_differences += sum(
                    int(float(after) != float(before))
                    for before, after in zip(before_rewards, after_rewards)
                )
                delta = float(event.get("graph_rpo_delta", 0.0))
                delta_sum += delta
                delta_abs_sum += abs(delta)
                nonzero_edit_count += int(abs(delta) > 1e-12)

    summary = {
        "files": len(files),
        "records": record_count,
        "unique_episodes": len(seen_episodes),
        "episodes_with_graph_trace": episodes_with_trace,
        "episodes_with_counterfactual_credit": episodes_with_credit,
        "counterfactual_edits": edit_count,
        "probe_responses": probe_count,
        "exactly_one_answer_tag": tagged_count,
        "answer_tag_rate": tagged_count / probe_count if probe_count else 0.0,
        "positive_probe_rewards": positive_count,
        "positive_probe_rate": positive_count / probe_count if probe_count else 0.0,
        "paired_sample_reward_differences": paired_reward_differences,
        "nonzero_edit_deltas": nonzero_edit_count,
        "delta_sum": delta_sum,
        "delta_abs_sum": delta_abs_sum,
        "integrity_errors": len(integrity_errors),
    }
    return {
        "summary": summary,
        "malformed_response_samples": malformed_samples,
        "integrity_error_samples": integrity_errors[:max_samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-tag-rate", type=float)
    parser.add_argument("--require-nonzero-delta", action="store_true")
    parser.add_argument("--fail-on-integrity-error", action="store_true")
    args = parser.parse_args()
    report = audit_results(args.inputs, max_samples=max(0, args.max_samples))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    summary = report["summary"]
    failed = False
    if args.fail_on_integrity_error and summary["integrity_errors"]:
        failed = True
    if args.min_tag_rate is not None and summary["answer_tag_rate"] < args.min_tag_rate:
        failed = True
    if args.require_nonzero_delta and summary["nonzero_edit_deltas"] == 0:
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
