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
MCP_BLOCK_PATTERN = re.compile(r"<use_mcp_tool>(.*?)</use_mcp_tool>", re.DOTALL)


def extract_tag(block: str, name: str) -> str | None:
    match = re.search(
        rf"<{re.escape(name)}>(.*?)</{re.escape(name)}>",
        block,
        re.DOTALL,
    )
    return match.group(1).strip() if match else None


def mcp_calls(content: str) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []
    for block in MCP_BLOCK_PATTERN.findall(content):
        server = extract_tag(block, "server_name")
        tool = extract_tag(block, "tool_name")
        arguments = extract_tag(block, "arguments")
        if server and tool and arguments is not None:
            calls.append((server, tool, arguments))
    return calls


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
    transition_examples: dict[str, list[str]] = defaultdict(list)
    mcp_tool_pairs: Counter[str] = Counter()
    mcp_calls_per_message: Counter[str] = Counter()
    malformed_mcp_messages = 0
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
            calls = mcp_calls(message["content"])
            if "<use_mcp_tool>" in message["content"] and not calls:
                malformed_mcp_messages += 1
            if calls:
                mcp_calls_per_message[str(len(calls))] += 1
                mcp_tool_pairs.update(f"{server}/{tool}" for server, tool, _ in calls)
        for index, message in enumerate(messages):
            if message["role"] != "assistant" or index + 1 >= len(messages):
                continue
            transition = f"assistant:{assistant_signature(message['content'])}->user"
            if len(transition_examples[transition]) < examples_per_signature:
                next_content = " ".join(messages[index + 1]["content"].split())[:300]
                assistant_content = " ".join(message["content"].split())[:300]
                transition_examples[transition].append(
                    f"ASSISTANT: {assistant_content} || USER: {next_content}"
                )
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
        "assistant_to_user_examples": dict(transition_examples),
        "mcp_tool_pairs": dict(mcp_tool_pairs.most_common()),
        "mcp_calls_per_message": dict(mcp_calls_per_message.most_common()),
        "malformed_mcp_messages": malformed_mcp_messages,
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
    console_summary = {
        key: summary[key]
        for key in (
            "schema_version",
            "rows",
            "invalid_message_rows",
            "role_transitions",
            "mcp_tool_pairs",
            "mcp_calls_per_message",
            "malformed_mcp_messages",
            "assistant_signatures",
            "final_assistant_signatures",
            "message_count",
        )
    }
    console_summary["assistant_plain_to_user_examples"] = summary[
        "assistant_to_user_examples"
    ].get("assistant:plain->user", [])
    console_summary["full_audit"] = str(output)
    print(json.dumps(console_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
