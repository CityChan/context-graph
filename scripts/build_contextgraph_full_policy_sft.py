#!/usr/bin/env python3
"""Curate native DeepSeek main and branch chats into complete-policy SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_contextgraph_sft import load_results, result_to_sft_row


FUNCTION_PATTERN = re.compile(r"<function=([^>]+)>")
BRANCH_MARKERS = ("MODE: BRANCH", "You are now a research branch")
BRANCH_ALLOWED_FUNCTIONS = {"search", "open_page", "return"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument("--allow-branch-without-open-page", action="store_true")
    return parser.parse_args()


def normalize_messages(messages):
    if not isinstance(messages, list):
        return []
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            return []
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return []
        normalized.append({"role": role, "content": content})
    return normalized


def branch_start(messages):
    for index, message in enumerate(messages):
        if message["role"] == "user" and any(marker in message["content"] for marker in BRANCH_MARKERS):
            return index
    return None


def branch_trajectory_to_row(
    trajectory,
    *,
    task_id: str,
    query_hash: str,
    teacher_model: str,
    require_open_page: bool,
    rejection_reasons: list[str] | None = None,
):
    def reject(reason):
        if rejection_reasons is not None:
            rejection_reasons.append(reason)

    messages = normalize_messages(trajectory.get("messages"))
    start = branch_start(messages)
    if not messages or start is None:
        reject("branch_prompt_missing")
        return None
    suffix = messages[start + 1 :]
    assistant_suffix = [message for message in suffix if message["role"] == "assistant"]
    if not assistant_suffix:
        reject("branch_no_assistant")
        return None
    function_counts = Counter(
        function
        for message in assistant_suffix
        for function in FUNCTION_PATTERN.findall(message["content"])
    )
    unsupported = sorted(set(function_counts).difference(BRANCH_ALLOWED_FUNCTIONS))
    if unsupported:
        reject("branch_unsupported_function:" + ",".join(unsupported))
        return None
    if function_counts["search"] < 1:
        reject("branch_no_search")
        return None
    if require_open_page and function_counts["open_page"] < 1:
        reject("branch_no_open_page")
        return None
    if function_counts["return"] != 1 or "<function=return>" not in assistant_suffix[-1]["content"]:
        reject("branch_no_terminal_return")
        return None
    for message in assistant_suffix:
        openings = message["content"].count("<function=")
        closings = message["content"].count("</function>")
        if openings != closings:
            reject("branch_malformed_function")
            return None
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    trajectory_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "messages": messages,
        "tools": [],
        "enable_thinking": False,
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "query_hash": query_hash,
        "domain": "search",
        "source": "local_vllm_contextgraph_teacher",
        "teacher_provider": "local_vllm",
        "teacher_model": teacher_model,
        "policy_role": "branch",
        "task_reward": 1.0,
        "protocol_valid": True,
        "search_calls": function_counts["search"],
        "open_page_calls": function_counts["open_page"],
        "return_calls": function_counts["return"],
        "branch_calls": 0,
        "finish_calls": 0,
    }


def result_query_hash(result):
    trajectories = result.get("agent_trajectories") or []
    main = next((item for item in trajectories if item.get("is_main")), None)
    messages = normalize_messages((main or {}).get("messages", result.get("messages")))
    query = next((message["content"] for message in messages if message["role"] == "user"), "")
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def function_counts(messages):
    return Counter(
        function
        for message in messages
        if message["role"] == "assistant"
        for function in FUNCTION_PATTERN.findall(message["content"])
    )


def result_eligible_for_branch_rows(result):
    if result.get("status") != "success":
        return False, "runner_failed"
    if float(result.get("task_reward", result.get("score", 0.0)) or 0.0) < 1.0:
        return False, "task_reward"
    if not bool(result.get("is_finish", False)):
        return False, "unfinished"
    stats = result.get("env_stats") or {}
    if bool(stats.get("overlong", False)) or bool(stats.get("hit_token_limit", False)):
        return False, "overlong"
    return True, None


def build_rows(results, *, teacher_model: str, require_open_page: bool):
    rows = {}
    counters = Counter()
    for result in results:
        counters["input_results"] += 1
        branch_parent_valid, branch_parent_reason = result_eligible_for_branch_rows(result)
        main_reasons = []
        main_row = result_to_sft_row(
            result,
            min_task_reward=1.0,
            min_valid_graph_ops=1,
            min_structural_graph_ops=1,
            max_invalid_graph_ops=0,
            require_finish=True,
            enable_thinking=False,
            teacher_provider="local_vllm",
            teacher_model=teacher_model,
            require_graph_trace=True,
            min_graph_quality_score=1.0,
            max_redundant_graph_ops=0,
            rejection_reasons=main_reasons,
        )
        if main_row is None:
            counters["rejected_result:" + (main_reasons[0] if main_reasons else "unknown")] += 1
        else:
            task_id = str(main_row["task_id"])
            query_hash = result_query_hash(result)
            main_function_counts = function_counts(main_row["messages"])
            main_row.update(
                {
                    "query_hash": query_hash,
                    "policy_role": "main",
                    "protocol_valid": True,
                    "search_calls": main_function_counts["search"],
                    "open_page_calls": main_function_counts["open_page"],
                    "return_calls": main_function_counts["return"],
                    "branch_calls": main_function_counts["branch"],
                    "finish_calls": main_function_counts["finish"],
                }
            )
            rows[main_row["trajectory_id"]] = main_row
            counters["accepted_main"] += 1
        if not branch_parent_valid:
            counters["rejected_branch_parent:" + str(branch_parent_reason)] += 1
            continue
        task_id = str(result.get("task_id", "unknown"))
        query_hash = result_query_hash(result)
        counters["accepted_branch_parent"] += 1
        trajectories = result.get("agent_trajectories") or []
        for trajectory in trajectories:
            if trajectory.get("is_main"):
                continue
            branch_reasons = []
            row = branch_trajectory_to_row(
                trajectory,
                task_id=task_id,
                query_hash=query_hash,
                teacher_model=teacher_model,
                require_open_page=require_open_page,
                rejection_reasons=branch_reasons,
            )
            if row is None:
                counters["rejected_branch:" + (branch_reasons[0] if branch_reasons else "unknown")] += 1
                continue
            if row["trajectory_id"] in rows:
                counters["duplicate_branch"] += 1
                continue
            rows[row["trajectory_id"]] = row
            counters["accepted_branch"] += 1
    return list(rows.values()), counters


def split_rows(rows, validation_fraction: float, seed: int):
    hashes = sorted({row["query_hash"] for row in rows})
    random.Random(seed).shuffle(hashes)
    count = round(len(hashes) * validation_fraction)
    validation_hashes = set(hashes[:count])
    return (
        [row for row in rows if row["query_hash"] not in validation_hashes],
        [row for row in rows if row["query_hash"] in validation_hashes],
    )


def main():
    args = parse_args()
    if not 0.0 <= args.validation_fraction < 1.0:
        raise SystemExit("--validation-fraction must be in [0, 1)")
    rows, counters = build_rows(
        load_results(args.inputs),
        teacher_model=args.teacher_model,
        require_open_page=not args.allow_branch_without_open_page,
    )
    if not rows or not counters["accepted_branch"]:
        raise SystemExit(f"no complete branch policy rows accepted: {dict(counters)}")
    train, validation = split_rows(rows, args.validation_fraction, args.seed)
    for values, raw_path in ((train, args.output), (validation, args.validation_output)):
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(values).to_parquet(path, index=False)
    tool_counts = {
        tool: sum(int(row[f"{tool}_calls"]) for row in rows)
        for tool in ("search", "open_page", "return", "branch", "finish")
    }
    missing_tools = [tool for tool, count in tool_counts.items() if count < 1]
    if missing_tools:
        raise SystemExit(
            "complete-policy coverage gate failed; no accepted calls for: "
            + ", ".join(missing_tools)
        )
    manifest = {
        "schema_version": "contextgraph.full_policy_sft.v1",
        **dict(counters),
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "query_overlap": len(
            {row["query_hash"] for row in train}
            & {row["query_hash"] for row in validation}
        ),
        "tool_counts": tool_counts,
        "output": args.output,
        "validation_output": args.validation_output,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
