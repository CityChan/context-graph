"""Read-only audit of completed ScienceWorld attempts; standard library only.

Works as a standalone script copied out of Git, without changing running checkouts.
Reports associations and evidence, not causal attribution from aggregate scores.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load(root, method):
    manifests, records = [], {}
    for shard in sorted(root.glob(method + "*")):
        if not (shard / "manifest.json").is_file():
            continue
        manifest = read(shard / "manifest.json")
        if manifest.get("method") != method:
            raise ValueError(f"Wrong method in {shard}")
        manifests.append(manifest)
        for path in sorted((shard / "instances").glob("*/result.json")):
            row = read(path)
            identity = row["task_id"]
            if identity not in manifest["task_ids"] or identity in records:
                raise ValueError(f"Unexpected/duplicate task: {identity}")
            # Saved absolute Vista paths are stale after syncing artifacts to Windows.
            row["_audit_attempt"] = str(path.parent / Path(row["attempt"]).name)
            records[identity] = row
    if not manifests:
        raise ValueError(f"No {method} manifests under {root}")
    selected = [identity for m in manifests for identity in m["task_ids"]]
    if len(selected) != len(set(selected)):
        raise ValueError("Overlapping shards")
    return manifests, records


def differences(left, right, prefix=""):
    result = {}
    for key in sorted(set(left) | set(right)):
        a, b = left.get(key), right.get(key)
        name = prefix + key
        if isinstance(a, dict) and isinstance(b, dict):
            result.update(differences(a, b, name + "."))
        elif a != b:
            result[name] = {"contextgraph": a, "foldagent": b}
    return result


def protocol_check(cm, fm):
    excluded = {"method", "config", "task_ids", "shard_index"}
    base = {k: v for k, v in cm[0].items() if k not in excluded}
    mismatches = []
    for method, manifests in (("contextgraph", cm), ("foldagent", fm)):
        for index, manifest in enumerate(manifests):
            diff = differences(base, {k: v for k, v in manifest.items() if k not in excluded})
            if diff:
                mismatches.append({"method": method, "shard": index, "fields": diff})
            if manifest.get("config") != manifests[0].get("config"):
                mismatches.append({"method": method, "shard": index, "error": "within-method config mismatch"})
    allowed = {"actor_rollout_ref.rollout.plugin." + key for key in
               ("workflow", "structured_graph_controller", "controller_owned_tool_formatting")}
    config_diff = differences(cm[0].get("config", {}), fm[0].get("config", {}))
    unexpected = {key: value for key, value in config_diff.items() if key not in allowed}
    same_selection = {t for m in cm for t in m["task_ids"]} == {t for m in fm for t in m["task_ids"]}
    return {"matched": not mismatches and not unexpected and same_selection,
            "mismatches": mismatches, "config_differences": config_diff,
            "unexpected_config_differences": unexpected, "same_selected_tasks": same_selection,
            "contextgraph_manifest": cm[0], "foldagent_manifest": fm[0]}


def aggregate(rows):
    graded = [r for r in rows if r["status"] == "graded"]
    keys = ("environment_steps", "environment_step_limit", "hit_max_turn", "hit_token_limit",
            "turn_budget_used", "graph_controller_turns", "graph_controller_counts_as_turn",
            "hit_timeout", "consol_attempts", "consol_controller_errors", "consol_invalid",
            "graph_explicit_ops", "controller_mode_rejections", "invalid_tool", "empty_command",
            "session_time", "memory_retrieval_calls", "observation_budget_skips")
    metrics = {}
    for key in keys:
        values = [r.get("env_stats", {})[key] for r in graded if key in r.get("env_stats", {})]
        metrics[key] = {"observed": len(values), "mean": mean(values) if values else None,
                        "positive": sum(v > 0 for v in values)}
    return {"completed": len(rows), "graded": len(graded),
            "infrastructure_errors": len(rows) - len(graded),
            "mean_score": mean(r["score"] for r in graded) if graded else None,
            "successes": sum(r["success"] for r in graded),
            "success_rate": mean(r["success"] for r in graded) if graded else None,
            "termination_reasons": dict(Counter(r.get("termination_reason") for r in graded)),
            "elapsed_seconds_mean": mean(r["elapsed_seconds"] for r in graded if "elapsed_seconds" in r)
                if any("elapsed_seconds" in r for r in graded) else None,
            "metrics": metrics}


def request_stats(path):
    stats = Counter()
    if not path.exists():
        return {"available": False}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids = row["output_ids"]
            stats["model_requests"] += 1
            stats["output_tokens"] += len(ids)
            stats["input_tokens"] += len(row["input_ids"])
            stats["length_finishes"] += row.get("finish_reason") == "length"
            run = maximum = 0
            for token in ids:
                run = run + 1 if token == 0 else 0
                maximum = max(maximum, run)
            stats["degenerate_requests"] += maximum >= 20
            schema = (row.get("structured_outputs") or {}).get("json", {})
            if isinstance(schema, dict) and "candidate_indices" in schema.get("properties", {}):
                stats["graph_controller_requests"] += 1
                stats["graph_controller_output_tokens"] += len(ids)
    return {"available": True, **stats}


def evidence(row, scan_requests):
    attempt = Path(row["_audit_attempt"])
    result = {"attempt": str(attempt), "saved_attempt": row["attempt"], "score": row["score"], "success": row["success"],
              "termination_reason": row.get("termination_reason"), "env_stats": row.get("env_stats", {})}
    tools = attempt / "tools.jsonl"
    steps = [json.loads(line) for line in tools.read_text(encoding="utf-8").splitlines() if line.strip()] if tools.exists() else []
    steps = [s for s in steps if s.get("event") == "step"]
    result["tool_log_available"] = tools.exists()
    result["steps_recorded"] = len(steps)
    result["repeated_adjacent_commands"] = sum(a.get("command") == b.get("command") for a, b in zip(steps, steps[1:]))
    result["top_commands"] = Counter(s.get("command") for s in steps).most_common(8)
    result["step_tail"] = [{"command": s.get("command"), "score": s.get("info", {}).get("score"),
                            "observation": str(s.get("observation", ""))[:800]} for s in steps[-8:]]
    trajectory = attempt / "trajectory.json"
    if trajectory.exists():
        trajectories = read(trajectory)
        main = next((r for r in trajectories if r.get("agent_name") == "main"), trajectories[0] if trajectories else {})
        result["num_branches"] = main.get("num_branches")
        result["message_tail"] = [{"role": m.get("role"), "content": str(m.get("content", ""))[:1200]}
                                  for m in main.get("messages", [])[-8:]]
    if scan_requests:
        result["requests"] = request_stats(attempt / "requests.jsonl")
    return result


def audit(croot, froot, case_limit=6, scan_requests=False):
    cm, cr = load(croot, "contextgraph")
    fm, fr = load(froot, "foldagent")
    common = sorted(t for t in cr.keys() & fr.keys() if cr[t]["status"] == fr[t]["status"] == "graded")
    if not common:
        raise ValueError("No commonly graded tasks")
    groups = {"foldagent_higher": [], "contextgraph_higher": [], "equal": []}
    by_type = defaultdict(list)
    for identity in common:
        delta = cr[identity]["score"] - fr[identity]["score"]
        groups["contextgraph_higher" if delta > 0 else "foldagent_higher" if delta < 0 else "equal"].append(identity)
        by_type[identity.removeprefix("scienceworld_test_").rsplit("_", 1)[0]].append(identity)
    def paired(ids):
        return {"count": len(ids), "contextgraph": aggregate([cr[t] for t in ids]),
                "foldagent": aggregate([fr[t] for t in ids]),
                "score_delta_contextgraph_minus_foldagent": mean(cr[t]["score"] - fr[t]["score"] for t in ids) if ids else None,
                "foldagent_only_success": sum(fr[t]["success"] and not cr[t]["success"] for t in ids),
                "contextgraph_only_success": sum(cr[t]["success"] and not fr[t]["success"] for t in ids)}
    cases = {}
    for group in ("foldagent_higher", "contextgraph_higher"):
        selected = sorted(groups[group], key=lambda t: (-abs(cr[t]["score"] - fr[t]["score"]), t))[:case_limit]
        for identity in selected:
            cases[identity] = {"group": group, "contextgraph": evidence(cr[identity], scan_requests),
                               "foldagent": evidence(fr[identity], scan_requests)}
    return {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol_check(cm, fm),
            "all_completed": {"contextgraph": aggregate(list(cr.values())), "foldagent": aggregate(list(fr.values()))},
            "common": paired(common), "common_task_ids": common,
            "groups": {key: paired(ids) for key, ids in groups.items()},
            "by_task_type": {key: paired(ids) for key, ids in sorted(by_type.items())},
            "cases": cases,
            "limits": ["Completed result files only; running attempts are excluded. Snapshot may advance during reading.",
                       "Common-task associations are not a causal ablation. Missing metrics are unknown, not zero.",
                       "Request/token-zero scans cover selected largest-gap cases only, not a whole-run quality certification."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contextgraph", required=True, type=Path)
    parser.add_argument("--foldagent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--scan-requests", action="store_true")
    args = parser.parse_args()
    if args.cases < 0:
        parser.error("cases must be nonnegative")
    for root in (args.contextgraph, args.foldagent):
        if args.output.resolve().is_relative_to(root.resolve()):
            parser.error("output must be outside both live run directories")
    report = audit(args.contextgraph, args.foldagent, args.cases, args.scan_requests)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"protocol_matched": report["protocol"]["matched"], "common": report["common"],
                      "by_task_type": {k: {"n": v["count"], "score_delta": v["score_delta_contextgraph_minus_foldagent"],
                                           "foldagent_only_success": v["foldagent_only_success"],
                                           "contextgraph_only_success": v["contextgraph_only_success"]}
                                       for k, v in report["by_task_type"].items()},
                      "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
