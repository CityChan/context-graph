#!/usr/bin/env python3
"""Audit persisted GraphRPO edit deltas and counterfactual probe integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BACKEND = "old_policy_counterfactual_qa"
SUPPORTED_BACKENDS = (
    "old_policy_counterfactual_qa",
    "reference_answer_likelihood",
    "old_policy_answer_likelihood",
    "external_evaluator",
)


if __package__:
    from .audit_records import input_files, load_records
else:
    from audit_records import input_files, load_records


def answer_format_valid(response: Any) -> bool:
    matches = re.findall(
        r"<answer>(.*?)</answer>",
        str(response or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return len(matches) == 1 and bool(matches[0].strip())


def semantic_state_hash(state: Any) -> str | None:
    """Hash graph content while ignoring bookkeeping-only counters."""
    if not isinstance(state, dict):
        return None
    semantic = {key: value for key, value in state.items() if key != "counters"}
    canonical = json.dumps(
        semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def semantic_change(event: dict[str, Any]) -> bool | None:
    """Return semantic-change status, or None for legacy compact traces."""
    before_hash = event.get("semantic_before_hash")
    after_hash = event.get("semantic_after_hash")
    if isinstance(before_hash, str) and isinstance(after_hash, str):
        return before_hash != after_hash
    before_hash = semantic_state_hash(event.get("before_state"))
    after_hash = semantic_state_hash(event.get("after_state"))
    if before_hash is not None and after_hash is not None:
        return before_hash != after_hash
    if event.get("before_hash") == event.get("after_hash"):
        return False
    return None


def credit_evaluation_status(event: dict[str, Any]) -> str:
    """Classify a selected event as scored, outcome-gated, or legacy-unknown."""
    if event.get("graph_rpo_outcome_gated") is True:
        return "outcome_gated"
    if event.get("graph_rpo_outcome_gated") is False:
        return "scored"
    if event.get("graph_rpo_delta_unclipped") is not None:
        return "scored"
    if event.get("graph_rpo_counterfactual_before_rewards") is not None:
        return "scored"
    return "unknown"


def quantile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated quantile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int]:
    """Summarize the raw credit scale without requiring NumPy."""
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p80": quantile(values, 0.80),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
    }


def audit_results(
    paths: Iterable[Path],
    max_samples: int = 20,
    backend: str = BACKEND,
) -> dict[str, Any]:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported GraphRPO backend: {backend}")
    files = input_files(paths)
    seen_episodes: set[str] = set()
    integrity_errors: list[dict[str, Any]] = []
    malformed_samples: list[dict[str, Any]] = []
    record_count = 0
    episodes_with_trace = 0
    episodes_with_credit = 0
    edit_count = 0
    scored_edit_count = 0
    outcome_gated_edit_count = 0
    evaluation_unknown_edit_count = 0
    semantic_change_count = 0
    semantic_noop_count = 0
    semantic_unknown_count = 0
    probe_count = 0
    tagged_count = 0
    positive_count = 0
    paired_reward_differences = 0
    nonzero_edit_count = 0
    positive_edit_count = 0
    negative_edit_count = 0
    zero_edit_count = 0
    zero_scored_edit_count = 0
    clipped_edit_count = 0
    delta_scales: list[float] = []
    delta_sum = 0.0
    delta_abs_sum = 0.0
    edits_by_op: Counter[str] = Counter()
    scored_edits_by_op: Counter[str] = Counter()
    outcome_gated_edits_by_op: Counter[str] = Counter()
    nonzero_edits_by_op: Counter[str] = Counter()
    delta_sum_by_op: defaultdict[str, float] = defaultdict(float)
    delta_abs_sum_by_op: defaultdict[str, float] = defaultdict(float)
    raw_deltas: list[float] = []
    raw_abs_deltas: list[float] = []
    raw_abs_deltas_by_op: defaultdict[str, list[float]] = defaultdict(list)
    clipped_edits_by_op: Counter[str] = Counter()
    semantic_noop_samples: list[dict[str, Any]] = []

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
                and event.get("graph_rpo_credit_backend") == backend
            ]
            if events:
                episodes_with_credit += 1

            for event in events:
                edit_count += 1
                seq = event.get("seq")
                op = str(event.get("op") or "unknown").lower()
                edits_by_op[op] += 1
                evaluation_status = credit_evaluation_status(event)
                if evaluation_status == "scored":
                    scored_edit_count += 1
                    scored_edits_by_op[op] += 1
                elif evaluation_status == "outcome_gated":
                    outcome_gated_edit_count += 1
                    outcome_gated_edits_by_op[op] += 1
                else:
                    evaluation_unknown_edit_count += 1
                changed = semantic_change(event)
                if changed is True:
                    semantic_change_count += 1
                elif changed is False:
                    semantic_noop_count += 1
                    if len(semantic_noop_samples) < max_samples:
                        semantic_noop_samples.append(
                            {
                                "source": str(path),
                                "gen_uid": gen_uid,
                                "seq": seq,
                                "op": op,
                            }
                        )
                else:
                    semantic_unknown_count += 1

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
                if backend == BACKEND:
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
                try:
                    numeric_delta_scale = float(event.get("graph_rpo_delta_scale"))
                except (TypeError, ValueError):
                    numeric_delta_scale = float("nan")
                if math.isfinite(numeric_delta_scale) and numeric_delta_scale > 0.0:
                    delta_scales.append(numeric_delta_scale)
                raw_delta = event.get("graph_rpo_delta_unclipped")
                scaled_delta = event.get(
                    "graph_rpo_delta_scaled_unclipped", raw_delta
                )
                delta_sum += delta
                delta_abs_sum += abs(delta)
                nonzero_edit_count += int(abs(delta) > 1e-12)
                positive_edit_count += int(delta > 1e-12)
                negative_edit_count += int(delta < -1e-12)
                zero_edit_count += int(abs(delta) <= 1e-12)
                zero_scored_edit_count += int(
                    evaluation_status == "scored" and abs(delta) <= 1e-12
                )
                edits_by_op[op] += 0
                nonzero_edits_by_op[op] += int(abs(delta) > 1e-12)
                delta_sum_by_op[op] += delta
                delta_abs_sum_by_op[op] += abs(delta)
                try:
                    numeric_raw_delta = float(raw_delta)
                    numeric_scaled_delta = float(scaled_delta)
                except (TypeError, ValueError):
                    continue
                if not (
                    math.isfinite(numeric_raw_delta)
                    and math.isfinite(numeric_scaled_delta)
                ):
                    continue
                if evaluation_status == "scored":
                    raw_deltas.append(numeric_raw_delta)
                    raw_abs_deltas.append(abs(numeric_raw_delta))
                    raw_abs_deltas_by_op[op].append(abs(numeric_raw_delta))
                was_clipped = abs(numeric_scaled_delta - delta) > 1e-12
                clipped_edit_count += int(was_clipped)
                clipped_edits_by_op[op] += int(was_clipped)

    summary = {
        "selected_backend": backend,
        "files": len(files),
        "records": record_count,
        "unique_episodes": len(seen_episodes),
        "episodes_with_graph_trace": episodes_with_trace,
        "episodes_with_counterfactual_credit": (
            episodes_with_credit if backend == BACKEND else 0
        ),
        "episodes_with_selected_credit": episodes_with_credit,
        "counterfactual_edits": edit_count if backend == BACKEND else 0,
        "selected_credit_edits": edit_count,
        "scored_credit_edits": scored_edit_count,
        "outcome_gated_edits": outcome_gated_edit_count,
        "credit_evaluation_unknown_edits": evaluation_unknown_edit_count,
        "semantic_state_change_edits": semantic_change_count,
        "semantic_noop_edits": semantic_noop_count,
        "semantic_unknown_edits": semantic_unknown_count,
        "probe_responses": probe_count,
        "exactly_one_answer_tag": tagged_count,
        "answer_tag_rate": tagged_count / probe_count if probe_count else 0.0,
        "positive_probe_rewards": positive_count,
        "positive_probe_rate": positive_count / probe_count if probe_count else 0.0,
        "paired_sample_reward_differences": paired_reward_differences,
        "nonzero_edit_deltas": nonzero_edit_count,
        "nonzero_edit_rate_all": (
            nonzero_edit_count / edit_count if edit_count else 0.0
        ),
        "nonzero_edit_rate": (
            nonzero_edit_count / scored_edit_count if scored_edit_count else 0.0
        ),
        "positive_edit_deltas": positive_edit_count,
        "negative_edit_deltas": negative_edit_count,
        "zero_edit_deltas": zero_edit_count,
        "zero_scored_edit_deltas": zero_scored_edit_count,
        "clipped_edit_deltas": clipped_edit_count,
        "clip_rate_all": clipped_edit_count / edit_count if edit_count else 0.0,
        "clip_rate": (
            clipped_edit_count / scored_edit_count if scored_edit_count else 0.0
        ),
        "delta_scale_distribution": distribution(delta_scales),
        "delta_sum": delta_sum,
        "delta_abs_sum": delta_abs_sum,
        "raw_delta_distribution": distribution(raw_deltas),
        "raw_abs_delta_distribution": distribution(raw_abs_deltas),
        "raw_abs_delta_distribution_by_op": {
            op: distribution(values)
            for op, values in sorted(raw_abs_deltas_by_op.items())
        },
        "edits_by_op": dict(sorted(edits_by_op.items())),
        "scored_edits_by_op": dict(sorted(scored_edits_by_op.items())),
        "outcome_gated_edits_by_op": dict(
            sorted(outcome_gated_edits_by_op.items())
        ),
        "nonzero_edit_deltas_by_op": dict(sorted(nonzero_edits_by_op.items())),
        "clipped_edit_deltas_by_op": dict(sorted(clipped_edits_by_op.items())),
        "delta_sum_by_op": dict(sorted(delta_sum_by_op.items())),
        "delta_abs_sum_by_op": dict(sorted(delta_abs_sum_by_op.items())),
        "integrity_errors": len(integrity_errors),
    }
    return {
        "summary": summary,
        "malformed_response_samples": malformed_samples,
        "semantic_noop_samples": semantic_noop_samples,
        "integrity_error_samples": integrity_errors[:max_samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=BACKEND)
    parser.add_argument("--min-tag-rate", type=float)
    parser.add_argument("--min-nonzero-rate", type=float)
    parser.add_argument("--max-clip-rate", type=float)
    parser.add_argument("--expected-delta-scale", type=float)
    parser.add_argument("--require-nonzero-delta", action="store_true")
    parser.add_argument("--fail-on-integrity-error", action="store_true")
    parser.add_argument("--fail-on-semantic-noop", action="store_true")
    args = parser.parse_args()
    report = audit_results(
        args.inputs,
        max_samples=max(0, args.max_samples),
        backend=args.backend,
    )
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
    if (
        args.min_nonzero_rate is not None
        and summary["nonzero_edit_rate"] < args.min_nonzero_rate
    ):
        failed = True
    if args.max_clip_rate is not None and summary["clip_rate"] > args.max_clip_rate:
        failed = True
    if args.expected_delta_scale is not None:
        expected = args.expected_delta_scale
        if not math.isfinite(expected) or expected <= 0.0:
            parser.error("--expected-delta-scale must be a positive finite number")
        observed = summary["delta_scale_distribution"]
        if (
            observed.get("count", 0) == 0
            or not math.isclose(
                float(observed["min"]), expected, rel_tol=1e-9, abs_tol=1e-12
            )
            or not math.isclose(
                float(observed["max"]), expected, rel_tol=1e-9, abs_tol=1e-12
            )
        ):
            failed = True
    if args.require_nonzero_delta and summary["nonzero_edit_deltas"] == 0:
        failed = True
    if args.fail_on_semantic_noop and summary["semantic_noop_edits"]:
        failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
