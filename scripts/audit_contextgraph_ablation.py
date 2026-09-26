"""Audit paired task outcomes and exact FoldAgent request equivalence."""
import argparse
import json
from collections import Counter
from pathlib import Path


def first_difference(left, right):
    """Zero-based mismatch, including a sequence ending before the other."""
    return next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                min(len(left), len(right)) if len(left) != len(right) else None)


def preview(value, limit=800):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return {"length": len(text), "text": text[:limit], "truncated": len(text) > limit}


def request_diagnostic(left, right):
    """Describe observable divergence without treating stored chat as raw output."""
    index = first_difference(left, right)
    result = {"first_different_request": index, "index_base": 0,
              "request_counts": [len(left), len(right)]}
    if index is None:
        result["category"] = "matching_requests"
        return result
    if index >= min(len(left), len(right)):
        result["category"] = "request_count_difference"
        return result
    a, b = left[index], right[index]
    result["different_fields"] = [k for k in sorted(set(a) | set(b))
                                  if (k in a) != (k in b) or a.get(k) != b.get(k)]
    result["other_fields"] = {k: {"foldagent": preview(a.get(k)), "equivalent": preview(b.get(k))}
                              for k in result["different_fields"] if k not in ("messages", "input_ids")}
    token_index = first_difference(a.get("input_ids", []), b.get("input_ids", []))
    result["tokens"] = {"first_difference": token_index,
                        "lengths": [len(a.get("input_ids", [])), len(b.get("input_ids", []))]}
    if token_index is not None:
        result["tokens"]["windows"] = [r.get("input_ids", [])[max(0, token_index - 8):token_index + 16]
                                        for r in (a, b)]
    messages_a, messages_b = a.get("messages", []), b.get("messages", [])
    message_index = first_difference(messages_a, messages_b)
    result["messages"] = {"first_difference": message_index,
                          "lengths": [len(messages_a), len(messages_b)]}
    if message_index is None:
        result["category"] = "same_messages_request_fields_differ"
        if "messages" not in a or "messages" not in b:
            result["category"] = "messages_unavailable"
        return result
    ma = messages_a[message_index] if message_index < len(messages_a) else None
    mb = messages_b[message_index] if message_index < len(messages_b) else None
    result["messages"].update({"foldagent": preview(ma), "equivalent": preview(mb)})
    if isinstance(ma, dict) and isinstance(mb, dict) and isinstance(ma.get("content"), str) and isinstance(mb.get("content"), str):
        char_index = first_difference(ma["content"], mb["content"])
        result["messages"]["first_content_character_difference"] = char_index
        if char_index is not None:
            result["messages"]["content_windows"] = [m["content"][max(0, char_index - 120):char_index + 400] for m in (ma, mb)]
    # Require append-only history on BOTH sides before calling this a new message.
    append_only = index > 0 and all(
        "messages" in r[index - 1] and "messages" in r[index]
        and len(r[index - 1]["messages"]) <= message_index
        and r[index]["messages"][:len(r[index - 1]["messages"])] == r[index - 1]["messages"]
        for r in (left, right))
    result["previous_request_equal"] = index > 0 and left[index - 1] == right[index - 1]
    roles = [m.get("role") if isinstance(m, dict) else None for m in (ma, mb)]
    result["messages"]["roles"] = roles
    if index == 0:
        result["category"] = "initial_messages_differ"
    elif append_only and roles == ["assistant", "assistant"]:
        result["category"] = "appended_assistant_message_differs"
    elif append_only:
        result["category"] = "appended_observation_or_message_differs"
    else:
        result["category"] = "history_rewrite_or_message_alignment_differs"
    return result


def task_diagnostic(task, left, right):
    main = request_diagnostic(left["model_contexts"], right["model_contexts"])
    a, b = left["branch_model_contexts"], right["branch_model_contexts"]
    return {"task_id": task, "main": main,
            "branch_keys_only_foldagent": sorted(set(a) - set(b)),
            "branch_keys_only_equivalent": sorted(set(b) - set(a)),
            "shared_branch_differences": {k: request_diagnostic(a[k], b[k])
                                          for k in sorted(set(a) & set(b)) if a[k] != b[k]}}


def read_main(directory):
    paths = list(Path(directory).glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"Expected one validation dump in {directory}, found {len(paths)}")
    rows = {}
    for line in paths[0].read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("agent_name") != "main":
            continue
        task = row.get("task_id")
        if not task or task == "unknown" or task in rows:
            raise ValueError(f"Missing or duplicate task identity: {task}")
        if not row.get("model_contexts"):
            raise ValueError(f"Missing model request audit for {task}")
        if not isinstance(row.get("branch_model_contexts"), dict):
            raise ValueError(f"Missing branch request audit for {task}")
        rows[task] = row
    return rows


def audit(root, peer_root=None):
    root = Path(root)
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    roots = [root]
    if peer_root is not None:
        peer_root = Path(peer_root)
        peer_manifest = json.loads((peer_root / "data/manifest.json").read_text(encoding="utf-8"))
        for key in ("commit", "source_sha256", "indices", "seed", "samples", "selection"):
            if key not in manifest or key not in peer_manifest or manifest[key] != peer_manifest[key]:
                raise ValueError(f"Paired manifest mismatch or missing field: {key}")
        roots.append(peer_root)
    expected = {f"bcp-diagnostic-{index}" for index in manifest["indices"]}
    variants = {}
    for name in ("foldagent", "equivalent", "legacy", "repaired"):
        locations = [directory / name for directory in roots if (directory / name).is_dir()]
        if len(locations) != 1:
            raise ValueError(f"Expected exactly one directory for variant {name}, found {len(locations)}")
        variants[name] = read_main(locations[0])
    for name, rows in variants.items():
        if set(rows) != expected:
            raise ValueError(f"Task coverage mismatch for {name}")
    scores = {}
    for name, rows in variants.items():
        successes = sum(float(row["task_reward"]) > 0 for row in rows.values())
        scores[name] = {"successes": successes, "samples": len(rows), "success_rate": successes / len(rows)}
    differences = []
    diagnostics = []
    for task in sorted(expected):
        left = variants["foldagent"][task]["model_contexts"]
        right = variants["equivalent"][task]["model_contexts"]
        # Request snapshots exclude random request IDs. Different generated
        # outputs can cause later divergence even with equivalent executors.
        first = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
        branches_match = variants["foldagent"][task].get("branch_model_contexts") == variants["equivalent"][task].get("branch_model_contexts")
        if first is not None or len(left) != len(right) or not branches_match:
            differences.append({"task_id": task, "first_different_request": first, "request_counts": [len(left), len(right)], "branches_match": branches_match})
            diagnostics.append(task_diagnostic(task, variants["foldagent"][task], variants["equivalent"][task]))
    result = {"commit": manifest["commit"], "run_roots": [str(p.resolve()) for p in roots], "scores": scores, "equivalence_request_differences": differences,
              "diagnostic_categories": dict(Counter(row["main"]["category"] for row in diagnostics)),
              "diagnostic_file": "equivalence_diagnostics.json",
              "note": "Paired diagnostics, not proof of a causal graph benefit. Request divergence requires inspection of preceding outputs."}
    detail = {"evaluation_commit": manifest["commit"], "request_index_base": 0,
              "limitations": "Messages are stored request history, not raw model completions. Assistant differences can reflect generation or postprocessing. Observation differences can reflect tools or orchestration. Snapshots omit some server-side settings. Branches are compared by stored key, not inferred semantic identity.",
              "tasks": diagnostics}
    (root / "equivalence_diagnostics.json").write_text(json.dumps(detail, indent=2, ensure_ascii=False), encoding="utf-8")
    (root / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--peer-root", help="Second run directory for a split four-variant evaluation")
    args = parser.parse_args()
    result = audit(args.root, args.peer_root)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["equivalence_request_differences"] else 0)
