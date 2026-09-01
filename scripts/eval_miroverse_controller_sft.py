#!/usr/bin/env python3
"""Evaluate a ContextGraph controller SFT model on held-out snapshots.

This evaluator deliberately bypasses the environment agent loop.  It renders
the held-out controller messages with the model tokenizer, performs guided JSON
generation with vLLM, and checks both the flat response schema and the
action-specific replay contract owned by :class:`GraphActionController`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph_controller import (
    GraphActionController,
    GraphControllerError,
    graph_action_schema,
)
from scripts.prepare_miroverse_contextgraph_controller_sft import (
    apply_resolved_action,
    build_graph,
)


REQUIRED_COLUMNS = {
    "messages",
    "candidate_count",
    "allow_pass",
    "action_policy",
    "action",
    "replay_valid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Merged Hugging Face model directory")
    parser.add_argument("--data", required=True, help="Held-out controller SFT parquet")
    parser.add_argument("--output", required=True, help="JSON summary output path")
    parser.add_argument("--max-samples", type=int, default=0, help="Rows to evaluate; 0 means all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def _as_messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        value = json.loads(value)
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError("messages is not a list")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("messages contains a non-object item")
        role = str(item.get("role", ""))
        content = item.get("content", "")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
            raise ValueError("messages contains an invalid role or content")
        messages.append({"role": role, "content": content})
    if len(messages) < 3 or messages[-1]["role"] != "assistant":
        raise ValueError("messages must end with the gold assistant decision")
    return messages


def _parse_json_object(response: str) -> dict[str, Any]:
    value = json.loads(response)
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def validate_flat_decision(
    decision: dict[str, Any],
    *,
    candidate_count: int,
    allow_pass: bool,
    action_policy: str,
) -> None:
    indices = list(range(candidate_count))
    schema = graph_action_schema(
        indices,
        allow_pass=allow_pass,
        action_policy=action_policy,
    )
    properties = schema["properties"]
    if set(decision) != set(schema["required"]):
        raise ValueError("response fields do not exactly match the controller schema")
    if decision["action"] not in properties["action"]["enum"]:
        raise ValueError("action is unavailable under the controller policy")
    selected = decision["candidate_indices"]
    if not isinstance(selected, list):
        raise ValueError("candidate_indices is not an array")
    if not 0 <= len(selected) <= min(6, candidate_count):
        raise ValueError("candidate_indices violates schema length bounds")
    if any(isinstance(index, bool) or not isinstance(index, int) or index not in indices for index in selected):
        raise ValueError("candidate_indices contains an unavailable index")
    if not isinstance(decision["summary"], str):
        raise ValueError("summary is not a string")
    if decision["relation"] not in properties["relation"]["enum"]:
        raise ValueError("relation is unavailable")


def replay_response(
    response: str,
    *,
    candidate_count: int,
    allow_pass: bool,
    action_policy: str,
) -> tuple[bool, str | None]:
    controller = GraphActionController(max_candidates=max(12, candidate_count))
    graph = build_graph(
        "Synthetic held-out controller replay question",
        [f"Synthetic evidence candidate {index}" for index in range(candidate_count)],
    )
    snapshot = controller.snapshot(graph)
    try:
        resolved = controller.resolve_action(
            graph,
            snapshot,
            response,
            allow_pass=allow_pass,
            action_policy=action_policy,
        )
        if not apply_resolved_action(graph, resolved):
            raise GraphControllerError("resolved action had no state effect")
    except (GraphControllerError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc)
    return True, None


def score_response(
    gold_response: str,
    predicted_response: str,
    *,
    candidate_count: int,
    allow_pass: bool,
    action_policy: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parse_valid": False,
        "schema_valid": False,
        "replay_valid": False,
        "action_exact": False,
        "indices_exact": False,
        "indices_set_exact": False,
        "relation_exact": False,
        "structural_choice_exact": False,
        "decision_exact": False,
        "predicted_action": None,
        "error": None,
    }
    try:
        gold = _parse_json_object(gold_response)
        predicted = _parse_json_object(predicted_response)
        result["parse_valid"] = True
        result["predicted_action"] = predicted.get("action")
        validate_flat_decision(
            predicted,
            candidate_count=candidate_count,
            allow_pass=allow_pass,
            action_policy=action_policy,
        )
        result["schema_valid"] = True
        replay_valid, replay_error = replay_response(
            predicted_response,
            candidate_count=candidate_count,
            allow_pass=allow_pass,
            action_policy=action_policy,
        )
        result["replay_valid"] = replay_valid
        result["error"] = replay_error
        result["action_exact"] = predicted.get("action") == gold.get("action")
        result["indices_exact"] = predicted.get("candidate_indices") == gold.get("candidate_indices")
        predicted_indices = predicted.get("candidate_indices")
        gold_indices = gold.get("candidate_indices")
        if isinstance(predicted_indices, list) and isinstance(gold_indices, list):
            result["indices_set_exact"] = set(predicted_indices) == set(gold_indices)
        result["relation_exact"] = predicted.get("relation") == gold.get("relation")
        result["structural_choice_exact"] = bool(
            result["action_exact"]
            and result["indices_set_exact"]
            and (gold.get("action") != "add_edge" or result["relation_exact"])
        )
        result["decision_exact"] = predicted == gold
    except (GraphControllerError, json.JSONDecodeError, TypeError, ValueError) as exc:
        result["error"] = str(exc)
    return result


def _rate(records: list[dict[str, Any]], key: str) -> float:
    return sum(bool(record[key]) for record in records) / len(records) if records else 0.0


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise SystemExit("--max-samples must be non-negative")
    data = pd.read_parquet(args.data)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise SystemExit(f"controller validation data is missing columns: {sorted(missing)}")
    data = data.loc[data["replay_valid"].astype(bool)].copy()
    if args.max_samples and len(data) > args.max_samples:
        data = data.sample(n=args.max_samples, random_state=args.seed)
    data = data.reset_index(drop=True)
    if data.empty:
        raise SystemExit("controller validation data has no replay-valid rows")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )
    prompts: list[str] = []
    gold_responses: list[str] = []
    groups: dict[tuple[int, bool, str], list[int]] = {}
    for row_index, row in data.iterrows():
        messages = _as_messages(row["messages"])
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(row.get("enable_thinking", False)),
        )
        prompts.append(prompt)
        gold_responses.append(messages[-1]["content"].strip())
        key = (int(row["candidate_count"]), bool(row["allow_pass"]), str(row["action_policy"]))
        groups.setdefault(key, []).append(row_index)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )
    predictions = [""] * len(data)
    for (candidate_count, allow_pass, action_policy), row_indices in sorted(groups.items()):
        schema = graph_action_schema(
            list(range(candidate_count)),
            allow_pass=allow_pass,
            action_policy=action_policy,
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=args.max_output_tokens,
            guided_decoding=GuidedDecodingParams(json=schema),
        )
        outputs = llm.generate([prompts[index] for index in row_indices], sampling, use_tqdm=True)
        for row_index, output in zip(row_indices, outputs, strict=True):
            predictions[row_index] = output.outputs[0].text.strip()

    scored: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for row_index, row in data.iterrows():
        score = score_response(
            gold_responses[row_index],
            predictions[row_index],
            candidate_count=int(row["candidate_count"]),
            allow_pass=bool(row["allow_pass"]),
            action_policy=str(row["action_policy"]),
        )
        scored.append(score)
        prediction_rows.append({
            "row_index": row_index,
            "trajectory_id": row.get("trajectory_id"),
            "task_id": row.get("task_id"),
            "query_hash": row.get("query_hash"),
            "gold_action": row["action"],
            "gold_response": gold_responses[row_index],
            "predicted_response": predictions[row_index],
            **score,
        })

    relation_rows = [
        score for score, (_, row) in zip(scored, data.iterrows(), strict=True)
        if row["action"] == "add_edge"
    ]
    summary = {
        "schema_version": "contextgraph.controller_eval.v1",
        "model": str(Path(args.model)),
        "data": str(Path(args.data)),
        "rows": len(scored),
        "parse_valid_rate": _rate(scored, "parse_valid"),
        "schema_valid_rate": _rate(scored, "schema_valid"),
        "action_contract_valid_rate": _rate(scored, "replay_valid"),
        "teacher_action_agreement": _rate(scored, "action_exact"),
        "teacher_candidate_indices_agreement": _rate(scored, "indices_exact"),
        "teacher_candidate_set_agreement": _rate(scored, "indices_set_exact"),
        "teacher_add_edge_relation_agreement": _rate(relation_rows, "relation_exact"),
        "teacher_structural_choice_agreement": _rate(scored, "structural_choice_exact"),
        "teacher_exact_decision_agreement": _rate(scored, "decision_exact"),
        "gold_action_counts": dict(Counter(str(action) for action in data["action"])),
        "predicted_action_counts": dict(Counter(str(score["predicted_action"]) for score in scored)),
        "error_counts": dict(Counter(str(score["error"]) for score in scored if score["error"])),
        "pass_gold_rows": int((data["action"] == "pass").sum()),
        "notes": [
            "teacher agreement is not unique semantic correctness because multiple graph actions may be legal",
            "action contract validity replays the predicted action on a synthetic graph with the same candidate topology",
            "pass behavior is not measured when pass_gold_rows is zero",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path = output.with_suffix(".predictions.parquet")
    pd.DataFrame(prediction_rows).to_parquet(predictions_path, index=False)
    print(json.dumps({**summary, "predictions": str(predictions_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
