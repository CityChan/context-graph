"""Route A: successful same-policy rollouts -> exact-context graph/control SFT.

Unlike the teacher curator, no model evidence reviewer or structural-edit gate
is required. Successful episodes without a model edit are retained as controls.
All loss-bearing tokens must come from an audited native completion request.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.exact_context import checksum, segments_from_requests, tokenizer_signature, validate_agent_context
from agents.graph_trace import validate_graph_trace
from scripts.build_auto_graph_sft import sampled_action, GRAPH_OPS
from scripts.build_bcp_graph_sft import read_json, read_jsonl, split_for
from scripts.generate_bcp_graph_sft import ROUTE_A_PROFILE, digest


def audit_episode(episode, trajectory, requests, tokenizer):
    from jsonschema.exceptions import ValidationError
    stats = trajectory.get("env_stats", {})
    if episode.get("status") != "ok" or episode.get("task_reward") != 1 or not episode.get("is_finish"):
        raise ValueError("task_not_solved")
    if stats.get("natural_finish") != 1 or any(stats.get(k, 0) for k in (
            "forced_finish", "hit_token_limit", "hit_max_turn", "hit_timeout", "invalid_tool",
            "action_protocol_errors", "graph_invalid_ops", "consol_invalid", "consol_controller_errors",
            "consol_pass_invalid", "judge_parse_failure", "env_error", "env_init_error")):
        raise ValueError("unclean_episode")
    if episode.get("benchmark") == "scienceworld" and (
            stats.get("completed") != 1 or stats.get("environment_natural_end") != 1):
        raise ValueError("environment_not_solved")
    if trajectory.get("graph_action_protocol") != "controller" or not stats.get("structured_graph_controller"):
        raise ValueError("wrong_protocol")
    valid, _, _ = validate_graph_trace(trajectory.get("graph_trace"))
    if not valid:
        raise ValueError("invalid_graph_trace")
    if not requests:
        raise ValueError("missing_requests")
    actions = []
    for request in requests:
        if request.get("native_token_ids") is not True:
            raise ValueError("native_token_ids_required")
        if tokenizer.decode(request["output_ids"], skip_special_tokens=True) != request["response"]:
            raise ValueError("output_token_text_mismatch")
        try:
            actions.append(sampled_action(request))
        except ValidationError as exc:
            raise ValueError("invalid_sampled_action_schema") from exc
    events = [e for e in trajectory["graph_trace"]["events"]
              if e.get("source") == "model" and e.get("op") in GRAPH_OPS]
    # Heuristic graph maintenance is never mislabeled as a sampled model edit.
    import hashlib
    sampled = Counter((a, hashlib.sha256(r["response"].encode()).hexdigest())
                      for a, r in zip(actions, requests) if a in GRAPH_OPS)
    actual = Counter((e["op"], e.get("assistant_content_sha256")) for e in events)
    if sampled != actual or any(not e.get("success") for e in events):
        raise ValueError("unmatched_graph_action")
    return "graph" if any(e["op"] != "pass" for e in events) else "control"


def mix_training(graph, control, ratio, seed):
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("control ratio must be finite and nonnegative")
    limit = math.floor(len(graph) * ratio)
    ordered = sorted(control, key=lambda r: checksum([seed, r["trajectory_id"]]))
    return sorted(graph + ordered[:limit], key=lambda r: checksum([seed, r["trajectory_id"], "mix"]))


def build(args, tokenizer):
    import pandas as pd
    root, output = Path(args.run), Path(args.output)
    manifest = read_json(root / "manifest.json")
    if manifest.get("profile") != ROUTE_A_PROFILE:
        raise ValueError("Expected Route A collection; teacher/test artifacts cannot be relabeled")
    settings = manifest["settings"]
    if Path(settings["tokenizer"]).resolve() != Path(args.tokenizer).resolve():
        raise ValueError("Use the collection checkpoint tokenizer")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty export directory")
    output.mkdir(parents=True, exist_ok=True)
    signature = tokenizer_signature(tokenizer)
    if signature != manifest.get("tokenizer_sha256"):
        raise ValueError("Collection tokenizer changed; refusing to relabel native tokens")
    counts, buckets, decisions, seen = Counter(), {}, [], set()
    expected = {(i, c) for i in manifest["indices"] for c in range(settings["candidates"])}
    for path in sorted(root.glob("episodes/*/episode.json")):
        episode = read_json(path)
        key = (episode["source_index"], episode["candidate"])
        if key not in expected or key in seen:
            raise ValueError("Unexpected/duplicate collected episode")
        seen.add(key)
        counts["episodes_seen"] += 1
        decision = {"episode": str(path.relative_to(root)), "task_id": episode["question_hash"]}
        try:
            trajectory = read_json(root / episode["trajectory"])
            requests = read_jsonl(root / episode["requests"])
            if any(r.get("requested_model") != settings["teacher_model"] for r in requests):
                raise ValueError("policy_model_mismatch")
            kind = audit_episode(episode, trajectory, requests, tokenizer)
            rows = segments_from_requests(requests, args.max_length, signature)
            split = split_for(episode["question_hash"], args.seed, args.validation_fraction)
            for row in rows:
                row.update(sft_profile="route_a_v1", task_id=episode["question_hash"],
                           source_index=episode["source_index"], candidate=episode["candidate"],
                           benchmark=episode["benchmark"], teacher_model=settings["teacher_model"],
                           enable_thinking=True, tools=[], kind=kind, split=split,
                           graph_action_protocol="controller", timing_source="framework",
                           task_reward=1.0, causal_graph_benefit_verified=False,
                           trajectory_id=checksum([key, row["agent_context"]["sha256"]]))
                validate_agent_context(row, args.max_length, signature)
            buckets.setdefault(f"{kind}_{split}", []).extend(rows)
            counts["accepted_episodes"] += 1
            counts[f"accepted:{kind}"] += 1
            decision.update(accepted=True, kind=kind, split=split, rows=len(rows))
        except (ValueError, KeyError, TypeError) as exc:
            counts["rejected:" + str(exc)] += 1
            decision.update(accepted=False, reason=str(exc))
        decisions.append(decision)
    counts["planned_episodes"], counts["incomplete_episodes"] = len(expected), len(expected - seen)
    buckets["train"] = mix_training(buckets.get("graph_train", []), buckets.get("control_train", []),
                                    args.control_ratio, args.seed)
    buckets["validation"] = buckets.get("graph_validation", []) + buckets.get("control_validation", [])
    hashes = {}
    for name, rows in buckets.items():
        if rows:
            target = output / (name + ".parquet")
            pd.DataFrame(rows).to_parquet(target, index=False)
            import hashlib
            hashes[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decisions), encoding="utf-8")
    report = {"profile": "route_a_v1", "source_manifest_sha256": digest(manifest),
              "settings": vars(args), "counts": dict(counts),
              "outputs": {k: len(v) for k, v in buckets.items() if v}, "files_sha256": hashes,
              "tokenizer_sha256": signature, "training_ready": bool(buckets["train"]),
              "validation_ready": bool(buckets["validation"]),
              "scope": "same-policy success self-distillation; framework timing; no causal edit claim"}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_complete and counts["incomplete_episodes"]:
        raise SystemExit("Collection incomplete; resume with the same manifest")
    if not buckets["train"] and not args.smoke:
        raise SystemExit("No graph training rows; inspect decisions (validation split is unchanged)")
    if args.smoke and not counts["accepted_episodes"]:
        raise SystemExit("No accepted Route A episodes")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--control-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.max_length < 1 or not 0 <= args.validation_fraction < 1:
        parser.error("Invalid length/split")
    from transformers import AutoTokenizer
    build(args, AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True))


if __name__ == "__main__":
    main()
