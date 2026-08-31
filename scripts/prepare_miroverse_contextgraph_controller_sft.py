#!/usr/bin/env python3
"""Convert MiroVerse trajectories into replay-verified controller SFT rows.

The smoke path intentionally emits one frozen ContextGraph decision per source
trajectory.  It uses the deployed GraphActionController prompt, schema, and
resolver so the generated supervision cannot silently drift from evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.context_graph import ContextGraph, EdgeRelation, NodeType
from agents.graph_controller import GraphActionController


TOOL_MARKERS = (
    "<use_mcp_tool>",
    "<search>",
    "<open_page>",
    "<browser",
    "<visit>",
    "<tool_call>",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="MiroVerse JSONL or parquet file")
    parser.add_argument("--output", required=True, help="Output controller SFT parquet")
    parser.add_argument("--manifest", help="Optional JSON manifest path")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--preview-chars", type=int, default=360)
    parser.add_argument("--teacher-base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--teacher-model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    parser.add_argument("--teacher-api-key", default="dummy")
    parser.add_argument("--teacher-timeout", type=float, default=600.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--source-subset", default="unknown")
    parser.add_argument(
        "--teacher-response",
        action="append",
        default=[],
        help="Offline JSON decision for deterministic tests; repeat once per sample",
    )
    return parser.parse_args()


def _as_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", item.get("from", ""))).strip().lower()
        content = item.get("content", item.get("value", ""))
        if role and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        yield from pd.read_parquet(path).to_dict(orient="records")
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _looks_like_tool_call(content: str) -> bool:
    lowered = content.lower()
    return any(marker in lowered for marker in TOOL_MARKERS)


def extract_evidence(record: dict[str, Any]) -> tuple[str, str, list[str]]:
    messages = _as_messages(record.get("messages", record.get("conversations")))
    question = str(record.get("query", "")).strip()
    if not question:
        question = next(
            (message["content"].strip() for message in messages if message["role"] == "user"),
            "",
        )
    answer = str(record.get("answer", record.get("raw_answer", ""))).strip()
    evidence: list[str] = []
    previous: dict[str, str] | None = None
    for message in messages:
        role, content = message["role"], message["content"].strip()
        if not content:
            previous = message
            continue
        if role in {"tool", "observation", "function"}:
            evidence.append(content)
        elif (
            role == "user"
            and previous is not None
            and previous["role"] == "assistant"
            and _looks_like_tool_call(previous["content"])
            and content != question
        ):
            evidence.append(content)
        previous = message
    # Preserve order while removing exact duplicate tool observations.
    evidence = list(dict.fromkeys(" ".join(item.split()) for item in evidence if item.strip()))
    return question, answer, evidence


def build_graph(question: str, evidence: list[str]) -> ContextGraph:
    graph = ContextGraph()
    root_id = graph.add_node(question, NodeType.QUERY)
    for observation in evidence:
        graph.add_node(
            observation[:800],
            NodeType.OBSERVATION,
            parent_id=root_id,
            edge_relation=EdgeRelation.CAUSAL,
            metadata={"source": "miroverse", "raw_content": observation},
        )
    return graph


def build_student_prompt(question: str, controller_prompt: str) -> str:
    return (
        "Original research question:\n"
        f"{question}\n\n"
        "Choose the best ContextGraph operation using only the frozen evidence "
        "candidates. The answer must obey the controller JSON schema.\n\n"
        f"{controller_prompt}"
    )


def build_teacher_prompt(student_prompt: str, gold_answer: str) -> str:
    private_context = (
        "\n\nPrivate teacher-only reference answer (never copy it into the action "
        f"summary unless supported by candidates):\n{gold_answer}"
        if gold_answer
        else ""
    )
    return student_prompt + private_context


def request_teacher(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    timeout: float,
    max_output_tokens: int,
) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You label ContextGraph controller actions. Select exactly one "
                    "legal, evidence-preserving action. Prefer add_edge for a real "
                    "directed dependency, merge only when the selected evidence can "
                    "be faithfully summarized together, and prune only when a "
                    "candidate is demonstrably irrelevant or redundant. Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_output_tokens,
        extra_body={
            "chat_template_kwargs": {"thinking": False},
            "structured_outputs": {"json": schema},
        },
    )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("teacher returned an empty response")
    return content.strip()


def apply_resolved_action(graph: ContextGraph, resolved: dict[str, Any]) -> bool:
    function = resolved["function"]
    arguments = resolved.get("arguments", {})
    if function == "pass":
        return True
    if function == "merge":
        return graph.merge(
            [item for item in str(arguments["node_ids"]).split(",") if item],
            str(arguments["summary"]),
        ) is not None
    if function == "prune":
        return graph.prune(str(arguments["node_id"]))
    if function == "select":
        return graph.select(str(arguments["node_id"]))
    if function == "add_edge":
        return graph.add_edge(
            str(arguments["source"]),
            str(arguments["target"]),
            EdgeRelation(str(arguments["relation"])),
        )
    raise ValueError(f"unsupported resolved action: {function}")


def convert_record(
    record: dict[str, Any],
    *,
    controller: GraphActionController,
    teacher_response: str,
    teacher_model: str,
    source_subset: str,
    sample_index: int,
) -> dict[str, Any]:
    question, _, evidence = extract_evidence(record)
    if not question:
        raise ValueError("trajectory has no query")
    if len(evidence) < 2:
        raise ValueError(f"trajectory has fewer than two tool observations: {len(evidence)}")
    graph = build_graph(question, evidence[-controller.max_candidates :])
    snapshot = controller.snapshot(graph)
    if len(snapshot.candidates) < 2:
        raise ValueError("controller snapshot has fewer than two candidates")
    allow_pass = graph.is_saturated()
    # Adding the final observation resets the idle-turn saturation clock. A
    # smoke snapshot should therefore require a productive structural action.
    allow_pass = bool(allow_pass and graph.turns_since_last_node_add >= 3)
    action_policy = "structural"
    controller_prompt = controller.action_prompt(
        snapshot,
        turn_id=len(evidence),
        allow_pass=allow_pass,
        action_policy=action_policy,
    )
    student_prompt = build_student_prompt(question, controller_prompt)
    before_hash = controller.graph_hash(graph)
    resolved = controller.resolve_action(
        graph,
        snapshot,
        teacher_response,
        allow_pass=allow_pass,
        action_policy=action_policy,
    )
    replay_graph = deepcopy(graph)
    if not apply_resolved_action(replay_graph, resolved):
        raise ValueError("resolved graph action had no state effect")
    after_hash = controller.graph_hash(replay_graph)
    if resolved["function"] != "pass" and after_hash == before_hash:
        raise ValueError("non-pass graph action did not change the graph hash")
    canonical_messages = [
        {
            "role": "system",
            "content": "You are the ContextGraph structural action controller.",
        },
        {"role": "user", "content": student_prompt},
        {"role": "assistant", "content": teacher_response},
    ]
    query_hash = hashlib.sha256(" ".join(question.lower().split()).encode("utf-8")).hexdigest()
    trajectory_id = hashlib.sha256(
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return {
        "messages": canonical_messages,
        "tools": [],
        "enable_thinking": False,
        "source": "miroverse_deepseek_v4_contextgraph_controller",
        "source_subset": str(record.get("split", source_subset)),
        "trajectory_id": trajectory_id,
        "task_id": str(record.get("id", sample_index)),
        "query_hash": query_hash,
        "checkpoint_turn": len(evidence),
        "graph_hash": snapshot.graph_hash,
        "candidate_count": len(snapshot.candidates),
        "candidate_snapshot_json": json.dumps(
            snapshot.trace_context(), ensure_ascii=False, sort_keys=True
        ),
        "allow_pass": allow_pass,
        "action_policy": action_policy,
        "action": resolved["function"],
        "teacher_response": teacher_response,
        "teacher_model": teacher_model,
        "replay_valid": True,
        "before_hash": before_hash,
        "after_hash": after_hash,
    }


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_suffix(".manifest.json")
    controller = GraphActionController(
        max_candidates=args.max_candidates,
        preview_chars=args.preview_chars,
    )
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    offline_responses = iter(args.teacher_response)
    for sample_index, record in enumerate(load_records(input_path)):
        if len(rows) >= args.max_samples:
            break
        try:
            question, answer, evidence = extract_evidence(record)
            if len(evidence) < 2:
                raise ValueError(f"trajectory has fewer than two tool observations: {len(evidence)}")
            graph = build_graph(question, evidence[-args.max_candidates :])
            snapshot = controller.snapshot(graph)
            allow_pass = False
            schema = controller.action_schema(
                snapshot,
                allow_pass=allow_pass,
                action_policy="structural",
            )
            controller_prompt = controller.action_prompt(
                snapshot,
                turn_id=len(evidence),
                allow_pass=allow_pass,
                action_policy="structural",
            )
            student_prompt = build_student_prompt(question, controller_prompt)
            try:
                teacher_response = next(offline_responses)
            except StopIteration:
                teacher_response = request_teacher(
                    base_url=args.teacher_base_url,
                    api_key=args.teacher_api_key,
                    model=args.teacher_model,
                    prompt=build_teacher_prompt(student_prompt, answer),
                    schema=schema,
                    timeout=args.teacher_timeout,
                    max_output_tokens=args.max_output_tokens,
                )
            rows.append(
                convert_record(
                    record,
                    controller=controller,
                    teacher_response=teacher_response,
                    teacher_model=args.teacher_model,
                    source_subset=args.source_subset,
                    sample_index=sample_index,
                )
            )
        except Exception as exc:
            rejected.append({"sample_index": sample_index, "reason": str(exc)})
    if len(rows) != args.max_samples:
        raise RuntimeError(
            f"smoke required {args.max_samples} accepted rows, got {len(rows)}; "
            f"rejections={rejected[:5]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    action_counts = pd.Series([row["action"] for row in rows]).value_counts().to_dict()
    manifest = {
        "schema_version": "contextgraph.controller_sft.v1",
        "input": str(input_path),
        "output": str(output_path),
        "accepted": len(rows),
        "rejected": len(rejected),
        "rejections": rejected,
        "action_counts": action_counts,
        "teacher_model": args.teacher_model,
        "source_subset": args.source_subset,
        "all_replay_valid": all(row["replay_valid"] for row in rows),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
