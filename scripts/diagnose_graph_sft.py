"""Inspect saved explicit SFT candidates without starting a teacher or simulator."""
import argparse
from collections import Counter
import json
from pathlib import Path


GRAPH_TOOLS = {"merge", "prune", "select", "add_edge", "graph_read", "graph_compact", "recall"}
STAT_KEYS = ("environment_steps", "environment_moves", "environment_score", "environment_peak_score",
             "completed", "natural_finish", "environment_natural_end", "environment_step_limit",
             "main_len", "main_context_tokens", "working_context_limit", "main_turn", "turn_budget_used",
             "hit_token_limit", "hit_max_turn", "hit_timeout", "session_time", "invalid_tool",
             "action_protocol_errors", "constrained_action_protocol", "graph_op_attempts",
             "graph_explicit_ops", "graph_context_refresh_tokens", "observation_budget_truncations",
             "observation_budget_skips")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def excerpt(value, limit=600):
    value = str(value)
    return value if len(value) <= limit else value[:limit // 2] + "\n...[excerpt]...\n" + value[-limit // 2:]


def schema_tools(schema):
    found = set()
    if isinstance(schema, dict):
        tool = schema.get("properties", {}).get("tool", {})
        if "const" in tool:
            found.add(tool["const"])
        for value in schema.values():
            found.update(schema_tools(value))
    elif isinstance(schema, list):
        for value in schema:
            found.update(schema_tools(value))
    return found


def diagnose(root):
    root = Path(root)
    manifest = read(root / "manifest.json")
    if manifest.get("profile") not in {"bcp_explicit_teacher_v1", "scienceworld_explicit_teacher_v1"}:
        raise ValueError("Expected an explicit graph SFT collection directory")
    settings = manifest["settings"]
    plugin = manifest.get("config", {}).get("actor_rollout_ref", {}).get("rollout", {}).get("plugin", {})
    tasks = []
    for path in sorted((root / "episodes").glob("*/episode.json")):
        episode = read(path)
        trajectory = read(root / episode["trajectory"])
        request_path = root / episode["requests"]
        counts, commands, endings = Counter(), Counter(), Counter()
        last, first, permitted = [], None, None
        malformed = 0
        for record in lines(request_path):
            if first is None:
                first = record
            allowed = schema_tools(record.get("schema", {}))
            permitted = allowed if permitted is None else permitted & allowed
            endings[str(record.get("finish_reason"))] += 1
            try:
                action = json.loads(record["response"])
                counts[action["tool"]] += 1
                command = action.get("arguments", {}).get("command")
                if command is not None:
                    commands[str(command)] += 1
            except (ValueError, KeyError, TypeError):
                malformed += 1
            last = (last + [record])[-2:]
        steps = []
        environment_init = None
        for event in lines(request_path.parent / "tools.jsonl"):
            if event.get("event") == "step":
                steps = (steps + [{k: excerpt(event[k]) if k == "observation" else event[k]
                                    for k in ("command", "observation", "score", "done") if k in event}])[-3:]
            elif environment_init is None:
                environment_init = {k: excerpt(v) for k, v in event.items()}
        stats = trajectory.get("env_stats", {})
        tasks.append({"source_index": episode["source_index"], "candidate": episode["candidate"],
                      "task_reward": episode.get("task_reward"), "is_finish": episode.get("is_finish"),
                      "stop": trajectory.get("termination_reason"),
                      "stats": {k: stats.get(k) for k in STAT_KEYS},
                      "request_count": sum(endings.values()), "finish_reasons": dict(endings),
                      "malformed_json_requests": malformed,
                      "tools_permitted_in_every_request": sorted(permitted) if permitted is not None else None,
                      "sampled_tools": dict(counts), "sampled_graph_tools": {k: counts[k] for k in GRAPH_TOOLS if counts[k]},
                      "most_common_commands": commands.most_common(6),
                      "environment_init": environment_init, "last_environment_steps": steps,
                      "initial_visible_messages": [{"role": m.get("role"), "content": excerpt(m.get("content"))}
                                                   for m in (first or {}).get("messages", [])],
                      "last_requests": [{"max_tokens": r.get("max_tokens"), "finish_reason": r.get("finish_reason"),
                                         "response": excerpt(r.get("response")),
                                         "last_visible_messages": [{"role": m.get("role"), "content": excerpt(m.get("content"))}
                                                                   for m in r.get("messages", [])[-2:]]} for r in last],
                      "requests_present": request_path.exists(), "requests_file": str(request_path)})
    return {"scope": "saved candidate diagnostics; no success or causal-benefit claim",
            "commit": manifest.get("commit"), "teacher": settings.get("teacher_model"),
            "settings": {k: settings.get(k) for k in ("benchmark", "samples", "candidates", "max_turn", "max_steps",
                                                      "prompt_length", "response_length", "branch_len", "turn_max_tokens")},
            "graph_config": {k: plugin.get(k) for k in ("graph_action_protocol", "action_output_profile",
                                                       "structured_graph_controller", "auto_prune_max_active",
                                                       "consolidation_interval", "initial_consolidation_turn")},
            "completed_candidates": len(tasks),
            "failed_attempts": sum(1 for _ in (root / "episodes").glob("*/attempt-*/error.json")),
            "tasks": tasks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(diagnose(args.run), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
        print(f"GRAPH_SFT_DIAGNOSTICS {args.output}")
    else:
        print(result)
