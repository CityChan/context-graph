#!/usr/bin/env python3
"""Inspect MiroVerse trajectories before converting them to policy SFT rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TAG_PATTERN = re.compile(r"<\s*/?\s*([A-Za-z_][A-Za-z0-9_.:-]*)")
FUNCTION_PATTERN = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_.-]*)>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    parser.add_argument("--examples-per-signature", type=int, default=2)
    return parser.parse_args()


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        import pandas as pd

        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def as_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        role = str(item.get("role", item.get("from", ""))).strip().lower()
        content = item.get("content", item.get("value", ""))
        if not role or not isinstance(content, str):
            return []
        messages.append({"role": role, "content": content})
    return messages


def assistant_signature(content: str) -> str:
    functions = FUNCTION_PATTERN.findall(content)
    if functions:
        return "function:" + "+".join(functions)
    tags = [tag.lower() for tag in TAG_PATTERN.findall(content)]
    if tags:
        return "tags:" + "+".join(dict.fromkeys(tags))
    return "plain"


def inspect_records(
    records: Iterable[dict[str, Any]],
    *,
    max_samples: int,
    examples_per_signature: int,
) -> dict[str, Any]:
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    record_keys: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    assistant_signatures: Counter[str] = Counter()
    final_assistant_signatures: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    message_counts: list[int] = []
    rows = 0
    invalid_messages = 0
    rows_with_query = 0
    rows_with_answer = 0
    rows_with_assistant = 0
    for record in records:
        if max_samples and rows >= max_samples:
            break
        rows += 1
        record_keys.update(str(key) for key in record)
        rows_with_query += bool(str(record.get("query", "")).strip())
        rows_with_answer += bool(
            str(record.get("answer", record.get("raw_answer", ""))).strip()
        )
        messages = as_messages(record.get("messages", record.get("conversations")))
        if not messages:
            invalid_messages += 1
            continue
        message_counts.append(len(messages))
        roles.update(message["role"] for message in messages)
        transitions.update(
            f"{left['role']}->{right['role']}"
            for left, right in zip(messages, messages[1:])
        )
        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        rows_with_assistant += bool(assistant_messages)
        for message in assistant_messages:
            signature = assistant_signature(message["content"])
            assistant_signatures[signature] += 1
            if len(examples[signature]) < examples_per_signature:
                examples[signature].append(" ".join(message["content"].split())[:500])
        if assistant_messages:
            final_assistant_signatures[assistant_signature(assistant_messages[-1]["content"])] += 1
    sorted_counts = sorted(message_counts)

    def percentile(fraction: float) -> int:
        if not sorted_counts:
            return 0
        index = round((len(sorted_counts) - 1) * fraction)
        return sorted_counts[index]

    return {
        "schema_version": "miroverse.policy_source_audit.v1",
        "rows": rows,
        "rows_with_query": rows_with_query,
        "rows_with_answer": rows_with_answer,
        "rows_with_assistant": rows_with_assistant,
        "invalid_message_rows": invalid_messages,
        "record_keys": dict(record_keys.most_common()),
        "roles": dict(roles.most_common()),
        "role_transitions": dict(transitions.most_common()),
        "assistant_signatures": dict(assistant_signatures.most_common()),
        "final_assistant_signatures": dict(final_assistant_signatures.most_common()),
        "assistant_examples": dict(examples),
        "message_count": {
            "min": sorted_counts[0] if sorted_counts else 0,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": sorted_counts[-1] if sorted_counts else 0,
        },
    }


def main() -> None:
    args = parse_args()
    summary = inspect_records(
        load_records(Path(args.input)),
        max_samples=args.max_samples,
        examples_per_signature=args.examples_per_signature,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
