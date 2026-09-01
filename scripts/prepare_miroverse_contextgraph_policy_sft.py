#!/usr/bin/env python3
"""Convert executable MiroVerse research trajectories to ContextGraph policy SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.prompts import create_chat
from scripts.inspect_miroverse_policy_source import as_messages, load_records, mcp_calls


SUPPORTED_TOOLS = {
    ("browsing-agent", "search_and_browse"): "branch",
    ("agent-browsing", "search_and_browse"): "branch",
    ("tool-google-search", "google_search"): "search",
    ("tool-google-search", "scrape"): "open_page",
}

FINALIZER_MARKERS = (
    "summarize the above conversation",
    "output the final answer",
    "final answer to the original question",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    parser.add_argument("--source-subset", default="MiroVerse-MuSiQue")
    return parser.parse_args()


def xml_value(value: Any) -> str:
    return str(value).replace("</parameter>", "</ parameter>").strip()


def branch_description(prompt: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt)
    return " ".join(words[:5]) or "research subtask"


def parse_arguments(raw: str) -> dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def convert_call(server: str, tool: str, raw_arguments: str) -> str | None:
    arguments = parse_arguments(raw_arguments)
    if arguments is None:
        return None
    mapped = SUPPORTED_TOOLS.get((server, tool))
    if mapped == "branch":
        prompt = str(arguments.get("subtask", "")).strip()
        if not prompt:
            return None
        return (
            "<function=branch>\n"
            f"<parameter=description>{xml_value(branch_description(prompt))}</parameter>\n"
            f"<parameter=prompt>{xml_value(prompt)}</parameter>\n"
            "</function>"
        )
    if mapped == "search":
        query = str(arguments.get("q", arguments.get("query", ""))).strip()
        if not query:
            return None
        topk = arguments.get("num", arguments.get("topk", 10))
        try:
            topk = max(1, min(20, int(topk)))
        except (TypeError, ValueError):
            topk = 10
        return (
            "<function=search>\n"
            f"<parameter=query>{xml_value(query)}</parameter>\n"
            f"<parameter=topk>{topk}</parameter>\n"
            "</function>"
        )
    if mapped == "open_page":
        url = str(
            arguments.get(
                "url",
                arguments.get("link", arguments.get("page_url", "")),
            )
        ).strip()
        docid = str(arguments.get("docid", arguments.get("document_id", ""))).strip()
        if url:
            parameter_name, parameter_value = "url", url
        elif docid:
            parameter_name, parameter_value = "docid", docid
        else:
            return None
        return (
            "<function=open_page>\n"
            f"<parameter={parameter_name}>{xml_value(parameter_value)}</parameter>\n"
            "</function>"
        )
    return None


def boxed_answer(text: str) -> str | None:
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    answer = text[start:index].strip()
                    return answer or None
    return None


def finish_call(final_text: str) -> str:
    answer = boxed_answer(final_text) or final_text.strip()
    return (
        "<function=finish>\n"
        f"<parameter=answer>{xml_value(answer)}</parameter>\n"
        f"<parameter=explanation>{xml_value(final_text)}</parameter>\n"
        "<parameter=confidence>100%</parameter>\n"
        "</function>"
    )


def strip_mcp_blocks(text: str) -> str:
    return re.sub(r"\s*<use_mcp_tool>.*?</use_mcp_tool>\s*", "", text, flags=re.DOTALL).strip()


def collapse_finalizer_round(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], bool]:
    if len(messages) < 5:
        return messages, False
    candidate_answer, finalizer_request, reformatted_answer = messages[-3:]
    request = finalizer_request["content"].lower()
    if (
        candidate_answer["role"] == "assistant"
        and finalizer_request["role"] == "user"
        and reformatted_answer["role"] == "assistant"
        and not mcp_calls(candidate_answer["content"])
        and not mcp_calls(reformatted_answer["content"])
        and any(marker in request for marker in FINALIZER_MARKERS)
    ):
        return messages[:-2], True
    return messages, False


def record_to_row(
    record: dict[str, Any],
    *,
    source_subset: str,
    rejection_reasons: list[str] | None = None,
) -> dict[str, Any] | None:
    def reject(reason: str) -> None:
        if rejection_reasons is not None:
            rejection_reasons.append(reason)

    source_messages = as_messages(record.get("messages", record.get("conversations")))
    source_messages, finalizer_collapsed = collapse_finalizer_round(source_messages)
    if len(source_messages) < 3 or source_messages[0]["role"] != "system":
        reject("message_shape")
        return None
    if source_messages[1]["role"] != "user" or source_messages[-1]["role"] != "assistant":
        reject("message_shape")
        return None
    query = source_messages[1]["content"].strip()
    if not query:
        reject("empty_query")
        return None

    canonical_messages = create_chat(
        query,
        "search_graph",
        expose_graph_tools=False,
    )
    mapped_counts: Counter[str] = Counter()
    for index, message in enumerate(source_messages[2:], start=2):
        is_final = index == len(source_messages) - 1
        if is_final:
            if (
                message["role"] != "assistant"
                or "<use_mcp_tool>" in message["content"]
                or "</use_mcp_tool>" in message["content"]
            ):
                reject("final_not_plain")
                return None
            canonical_messages.append({"role": "assistant", "content": finish_call(message["content"])})
            continue
        expected_role = "assistant" if index % 2 == 0 else "user"
        if message["role"] != expected_role:
            reject("role_alternation")
            return None
        if message["role"] == "user":
            canonical_messages.append(message)
            continue
        calls = mcp_calls(message["content"])
        if (
            message["content"].count("<use_mcp_tool>") != len(calls)
            or message["content"].count("</use_mcp_tool>") != len(calls)
        ):
            reject("malformed_mcp_call")
            return None
        if len(calls) != 1:
            reject("assistant_call_count")
            return None
        server, tool, raw_arguments = calls[0]
        if (server, tool) not in SUPPORTED_TOOLS:
            reject(f"unsupported_tool:{server}/{tool}")
            return None
        converted = convert_call(server, tool, raw_arguments)
        if converted is None:
            reject(f"invalid_tool_arguments:{server}/{tool}")
            return None
        reasoning = strip_mcp_blocks(message["content"])
        content = f"{reasoning}\n\n{converted}" if reasoning else converted
        canonical_messages.append({"role": "assistant", "content": content})
        mapped_counts[SUPPORTED_TOOLS[(server, tool)]] += 1

    if not mapped_counts:
        reject("no_tool_calls")
        return None
    canonical = json.dumps(
        canonical_messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    trajectory_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return {
        "messages": canonical_messages,
        "tools": [],
        "enable_thinking": False,
        "trajectory_id": trajectory_id,
        "task_id": query_hash,
        "query_hash": query_hash,
        "domain": "search",
        "source": "miroverse_contextgraph_policy",
        "source_subset": source_subset,
        "policy_protocol": "contextgraph.environment_xml.v1",
        "protocol_valid": True,
        "policy_turns": sum(message["role"] == "assistant" for message in canonical_messages),
        "mapped_branch_calls": mapped_counts["branch"],
        "mapped_search_calls": mapped_counts["search"],
        "mapped_open_page_calls": mapped_counts["open_page"],
        "source_finalizer_collapsed": finalizer_collapsed,
    }


def build_rows(
    records: Iterable[dict[str, Any]],
    *,
    max_samples: int,
    source_subset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    rows_by_id: dict[str, dict[str, Any]] = {}
    counters: dict[str, Any] = {
        "input": 0,
        "accepted": 0,
        "duplicates": 0,
        "rejected": 0,
        "rejection_reasons": {},
    }
    reasons_counter: Counter[str] = Counter()
    for record in records:
        if max_samples and counters["input"] >= max_samples:
            break
        counters["input"] += 1
        reasons: list[str] = []
        row = record_to_row(
            record,
            source_subset=source_subset,
            rejection_reasons=reasons,
        )
        if row is None:
            counters["rejected"] += 1
            reasons_counter[reasons[0] if reasons else "unknown"] += 1
            continue
        if row["trajectory_id"] in rows_by_id:
            counters["duplicates"] += 1
            continue
        rows_by_id[row["trajectory_id"]] = row
        counters["accepted"] += 1
    counters["rejection_reasons"] = dict(reasons_counter.most_common())
    return list(rows_by_id.values()), counters


def split_rows(
    rows: list[dict[str, Any]],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query_hashes = sorted({row["query_hash"] for row in rows})
    random.Random(seed).shuffle(query_hashes)
    validation_count = round(len(query_hashes) * validation_fraction)
    validation_hashes = set(query_hashes[:validation_count])
    train = [row for row in rows if row["query_hash"] not in validation_hashes]
    validation = [row for row in rows if row["query_hash"] in validation_hashes]
    return train, validation


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in [0, 1)")
    rows, counters = build_rows(
        load_records(Path(args.input)),
        max_samples=args.max_samples,
        source_subset=args.source_subset,
    )
    if not rows:
        raise SystemExit(f"no executable policy rows accepted: {counters}")
    train, validation = split_rows(
        rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    output = Path(args.output)
    validation_output = Path(args.validation_output)
    write_parquet(train, output)
    write_parquet(validation, validation_output)
    manifest = {
        "schema_version": "contextgraph.policy_sft.v1",
        "input": args.input,
        "source_subset": args.source_subset,
        **counters,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "query_overlap": len(
            {row["query_hash"] for row in train}
            & {row["query_hash"] for row in validation}
        ),
        "mapped_tool_counts": dict(
            Counter(
                {
                    "branch": sum(row["mapped_branch_calls"] for row in rows),
                    "search": sum(row["mapped_search_calls"] for row in rows),
                    "open_page": sum(row["mapped_open_page_calls"] for row in rows),
                }
            )
        ),
        "collapsed_finalizer_rows": sum(
            bool(row["source_finalizer_collapsed"]) for row in rows
        ),
        "output": str(output),
        "validation_output": str(validation_output),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
