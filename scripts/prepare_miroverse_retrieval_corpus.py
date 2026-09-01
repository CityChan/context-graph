#!/usr/bin/env python3
"""Build a deduplicated local retrieval corpus from MiroVerse tool observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from scripts.inspect_miroverse_policy_source import as_messages, load_records, mcp_calls


SEARCH_TOOL_PAIRS = {
    ("browsing-agent", "search_and_browse"),
    ("agent-browsing", "search_and_browse"),
    ("tool-google-search", "google_search"),
    ("tool-google-search", "scrape"),
}
URL_PATTERN = re.compile(r"https?://[^\s<>\]\[\)\(\"']+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    parser.add_argument("--min-characters", type=int, default=40)
    parser.add_argument("--max-characters", type=int, default=100_000)
    return parser.parse_args()


def parse_arguments(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def normalize_observation(text: str) -> str:
    return " ".join(text.split())


def observation_title(server: str, tool: str, arguments: dict[str, Any]) -> str:
    for key in ("subtask", "q", "query", "title"):
        value = str(arguments.get(key, "")).strip()
        if value:
            return " ".join(value.split())[:300]
    url = str(
        arguments.get("url", arguments.get("link", arguments.get("page_url", "")))
    ).strip()
    return url[:300] if url else f"{server}/{tool} observation"


def observation_url(
    text: str,
    arguments: dict[str, Any],
    *,
    docid: str,
) -> tuple[str, bool]:
    for key in ("url", "link", "page_url"):
        value = str(arguments.get(key, "")).strip()
        if value.startswith(("http://", "https://")):
            return value, False
    match = URL_PATTERN.search(text)
    if match:
        return match.group(0).rstrip(".,;:"), False
    return f"miroverse://observation/{docid}", True


def build_corpus(
    records: Iterable[dict[str, Any]],
    *,
    max_samples: int,
    min_characters: int,
    max_characters: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if min_characters < 1 or max_characters < min_characters:
        raise ValueError("invalid observation character bounds")

    counters: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    unsupported_tool_counts: Counter[str] = Counter()
    documents: dict[str, dict[str, Any]] = {}

    for source_index, record in enumerate(records):
        if max_samples and counters["input_rows"] >= max_samples:
            break
        counters["input_rows"] += 1
        messages = as_messages(record.get("messages", record.get("conversations")))
        if not messages:
            counters["invalid_message_rows"] += 1
            continue
        counters["rows_with_messages"] += 1

        for turn_index, message in enumerate(messages[:-1]):
            if message["role"] != "assistant":
                continue
            calls = mcp_calls(message["content"])
            counters["mcp_calls_seen"] += len(calls)
            if not calls:
                continue
            next_message = messages[turn_index + 1]
            if next_message["role"] != "user":
                counters["missing_user_observation"] += len(calls)
                continue
            observation = normalize_observation(next_message["content"])
            supported_calls = [call for call in calls if call[:2] in SEARCH_TOOL_PAIRS]
            for server, tool, _ in calls:
                pair = f"{server}/{tool}"
                if (server, tool) not in SEARCH_TOOL_PAIRS:
                    unsupported_tool_counts[pair] += 1
            if not supported_calls:
                continue
            counters["supported_tool_calls"] += len(supported_calls)
            counters["observation_messages"] += 1
            if len(observation) < min_characters:
                counters["rejected_short_observation"] += 1
                continue
            if len(observation) > max_characters:
                observation = observation[:max_characters].rstrip()
                counters["truncated_observations"] += 1

            content_hash = hashlib.sha256(observation.encode("utf-8")).hexdigest()
            if content_hash in documents:
                documents[content_hash]["duplicate_count"] += 1
                counters["duplicate_observations"] += 1
                continue

            server, tool, raw_arguments = supported_calls[0]
            arguments = parse_arguments(raw_arguments)
            docid = f"miroverse_obs_{content_hash[:24]}"
            url, synthetic_url = observation_url(
                observation,
                arguments,
                docid=docid,
            )
            documents[content_hash] = {
                "docid": docid,
                "url": url,
                "title": observation_title(server, tool, arguments),
                "text": observation,
                "source": "miromind-ai/MiroVerse-v0.1:MiroVerse-MuSiQue",
                "source_index": source_index,
                "turn_index": turn_index,
                "server_name": server,
                "tool_name": tool,
                "content_hash": content_hash,
                "duplicate_count": 0,
            }
            counters["synthetic_urls"] += int(synthetic_url)
            for call_server, call_tool, _ in supported_calls:
                tool_counts[f"{call_server}/{call_tool}"] += 1

    rows = sorted(documents.values(), key=lambda row: row["docid"])
    counters["accepted_documents"] = len(rows)
    return rows, {
        "counters": dict(counters),
        "tool_counts": dict(tool_counts.most_common()),
        "unsupported_tool_counts": dict(unsupported_tool_counts.most_common()),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    rows, audit = build_corpus(
        load_records(input_path),
        max_samples=args.max_samples,
        min_characters=args.min_characters,
        max_characters=args.max_characters,
    )
    if not rows:
        raise SystemExit(f"no MiroVerse retrieval documents accepted: {audit}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    manifest = {
        "schema_version": "miroverse.retrieval_corpus.v1",
        "input": str(input_path),
        "output": str(output_path),
        "source": "miromind-ai/MiroVerse-v0.1:MiroVerse-MuSiQue",
        "deduplication": "sha256(normalized_observation_text)",
        **audit,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
