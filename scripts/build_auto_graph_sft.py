"""Curate actual auto-controller requests, with only the sampled last turn supervised.

Automatic history replacement changes old conditioning prefixes. Those historical
assistant messages are input only, never relabeled as freshly sampled responses.
Heuristic edits remain in the graph trace, never become assistant targets.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph_trace import validate_graph_trace
from scripts.generate_bcp_graph_sft import AUTO_PROFILE, TeacherClient, digest
from scripts.build_bcp_graph_sft import read_json, read_jsonl, review_edits, split_for

GRAPH_OPS = {"merge", "prune", "add_edge", "select", "pass"}


def sampled_action(record):
    """Validate the actual transport response; never reconstruct an executor call."""
    import jsonschema
    from agents.agent_text import extract_fn_call
    if record.get("finish_reason") != "stop" or not record.get("messages"):
        raise ValueError("incomplete_request")
    schema = record.get("schema")
    if schema is not None:
        payload = json.loads(record["response"])
        jsonschema.validate(payload, schema)
        if payload.get("action") not in GRAPH_OPS:
            raise ValueError("unknown_controller_action")
        return payload["action"]
    call = extract_fn_call(record["response"])
    if call is None:
        raise ValueError("invalid_tool_response")
    return call["function"]


def audit_auto_episode(episode, trajectory, requests):
    stats = trajectory.get("env_stats", {})
    if episode.get("status") != "ok" or episode.get("task_reward") != 1 or not episode.get("is_finish"):
        return "task_not_solved", None
    if episode.get("benchmark") == "scienceworld" and (
            stats.get("completed") != 1 or stats.get("environment_natural_end") != 1):
        return "environment_not_solved", None
    if trajectory.get("graph_action_protocol") != "controller" or not stats.get("structured_graph_controller"):
        return "wrong_protocol", None
    if stats.get("natural_finish") != 1 or any(stats.get(k, 0) for k in (
            "forced_finish", "graph_invalid_ops", "consol_controller_errors", "consol_invalid",
            "consol_pass_invalid", "action_protocol_errors", "judge_parse_failure", "env_error",
            "env_init_error", "hit_token_limit", "hit_max_turn", "hit_timeout")):
        return "unclean_episode", None
    valid, errors, metrics = validate_graph_trace(trajectory.get("graph_trace"))
    if not valid:
        return "invalid_graph_trace", None
    if metrics["redundant_model_ops"]:
        return "redundant_graph_ops", None
    if not requests:
        return "missing_requests", None
    try:
        actions = [sampled_action(r) for r in requests]
    except Exception:
        return "invalid_sampled_action", None
    events = trajectory["graph_trace"]["events"]
    model_edits = [e for e in events if e.get("source") == "model" and e.get("op") in GRAPH_OPS]
    if not any(e["op"] in {"merge", "prune", "add_edge"} and e.get("success") for e in model_edits):
        return "no_model_structural_graph_action", None
    actual = Counter((e["op"], e.get("assistant_content_sha256")) for e in model_edits)
    sampled = Counter((a, hashlib.sha256(r["response"].encode()).hexdigest())
                      for r, a in zip(requests, actions) if a in GRAPH_OPS)
    if actual != sampled or any(not e.get("success") for e in model_edits):
        return "unmatched_graph_action", None
    if not any(a not in GRAPH_OPS for a in actions[max(i for i, a in enumerate(actions) if a in GRAPH_OPS) + 1:]):
        return "no_post_graph_continuation", None
    return None, {"events": model_edits, "actions": actions, "metrics": metrics,
                  "events_by_source": dict(Counter(str(e["source"]) + ":" + e["op"] for e in events))}


def request_row(record, episode, action, teacher):
    messages = record["messages"] + [{"role": "assistant", "content": record["response"]}]
    return {"messages": messages, "tools": [], "enable_thinking": False,
            "train_last_response_only": True, "trajectory_id": digest(messages),
            "task_id": episode["question_hash"], "source_index": episode["source_index"],
            "candidate": episode["candidate"], "benchmark": episode["benchmark"],
            "teacher_model": teacher, "graph_action_protocol": "controller", "task_reward": 1.0,
            "target_action": action, "target_role": "controller" if record.get("schema") else "actor",
            "timing_source": "framework" if record.get("schema") else "model",
            "graph_trace_valid": True, "causal_graph_benefit_verified": False}


async def build(args, tokenizer):
    import pandas as pd
    from omegaconf import OmegaConf
    root, output = Path(args.run), Path(args.output)
    manifest = read_json(root / "manifest.json")
    if manifest.get("profile") != AUTO_PROFILE:
        raise ValueError("Expected auto-controller collection manifest")
    output.mkdir(parents=True, exist_ok=False)
    settings = SimpleNamespace(**manifest["settings"])
    settings.temperature = 0.0
    config = OmegaConf.create(manifest["config"]).actor_rollout_ref.rollout
    reviewer = TeacherClient(settings, tokenizer, config, output / "review-requests.jsonl") if args.review else None
    counts, buckets, decisions, seen = Counter(), {}, [], set()
    paths = sorted((root / "episodes").glob("*/episode.json"))
    counts["planned_episodes"] = len(manifest["indices"]) * settings.candidates
    counts["incomplete_episodes"] = max(0, counts["planned_episodes"] - len(paths))
    counts["failed_attempts"] = sum(1 for _ in (root / "episodes").glob("*/attempt-*/error.json"))
    try:
        for path in paths:
            episode = read_json(path)
            trajectory = read_json(root / episode["trajectory"])
            requests = read_jsonl(root / episode["requests"])
            reason, audit = audit_auto_episode(episode, trajectory, requests)
            counts["episodes_seen"] += 1
            if not reason and any(e["op"] in {"merge", "prune"} for e in audit["events"]):
                reason = await review_edits(audit["events"], reviewer, tokenizer, args.review_max_input) if reviewer else "evidence_review_required"
            decision = {"episode": str(path.relative_to(root)), "reason": reason}
            decisions.append(decision)
            if reason:
                counts["rejected:" + reason] += 1
                continue
            counts["accepted_episodes"] += 1
            decision["events_by_source"] = audit["events_by_source"]
            split = split_for(episode["question_hash"], args.seed, args.validation_fraction)
            for record, action in zip(requests, audit["actions"]):
                row = request_row(record, episode, action, settings.teacher_model)
                if row["trajectory_id"] in seen:
                    continue
                length = len(tokenizer.apply_chat_template(row["messages"], tokenize=True, enable_thinking=False))
                if length > args.max_length:
                    counts["rejected_row:too_long"] += 1
                    continue
                seen.add(row["trajectory_id"])
                row.update(tokens=length, episode=str(path.relative_to(root)))
                bucket = row["target_role"] + "_" + split
                buckets.setdefault(bucket, []).append(row)
                counts["rows:" + bucket] += 1
                if action in {"merge", "prune", "add_edge"}:
                    counts["structural_rows:" + split] += 1
    finally:
        if reviewer:
            await reviewer.close()
    for name, rows in buckets.items():
        pd.DataFrame(rows).to_parquet(output / (name + ".parquet"), index=False)
    (output / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decisions), encoding="utf-8")
    report = {"counts": dict(counts), "outputs": {k: len(v) for k, v in buckets.items()},
              "source_manifest_sha256": digest(manifest), "settings": vars(args),
              "scope": "successful auto-controller demonstrations; framework timing; no autonomous timing or causal benefit claim",
              "training": "MultiTurnSFTDataset; train_last_response_only=True; no input truncation",
              "reviewer": settings.teacher_model if args.review else None}
    report['curation_smoke_only'] = bool(getattr(args, 'smoke', False))
    report['training_ready'] = bool(counts['rows:actor_train'] and counts['structural_rows:train'])
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if getattr(args, 'smoke', False):
        from scripts.audit_auto_graph_sft import audit_export
        smoke = audit_export(output)
        (output / 'smoke-audit.json').write_text(json.dumps(smoke, indent=2), encoding='utf-8')
        print(json.dumps(smoke, indent=2))
        if not smoke['passed']:
            raise SystemExit('Auto graph collection smoke failed; inspect decisions.jsonl and smoke-audit.json.')
        return
    if not counts["rows:controller_train"] or not counts["structural_rows:train"]:
        if counts['structural_rows:validation']:
            raise SystemExit('Qualified graph rows exist in validation only; training set is empty. Fixed task split preserved.')
        raise SystemExit("No qualified auto graph training rows; inspect decisions.jsonl. Gates were not relaxed.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--output", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--review", action="store_true")
    p.add_argument("--review-max-input", type=int, default=24000)
    p.add_argument('--smoke', action='store_true', help='Check qualified exports in either split; report training readiness separately')
    args = p.parse_args()
    if not 0 <= args.validation_fraction < 1 or min(args.max_length, args.review_max_input) < 1:
        p.error("Invalid curation lengths or split")
    from transformers import AutoTokenizer
    asyncio.run(build(args, AutoTokenizer.from_pretrained(args.tokenizer)))


if __name__ == "__main__":
    main()
