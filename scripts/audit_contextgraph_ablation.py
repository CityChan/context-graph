"""Audit paired task outcomes and exact FoldAgent request equivalence."""
import argparse
import json
from pathlib import Path


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


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    expected = {f"bcp-diagnostic-{index}" for index in manifest["indices"]}
    variants = {name: read_main(root / name) for name in ("foldagent", "equivalent", "legacy", "repaired")}
    for name, rows in variants.items():
        if set(rows) != expected:
            raise ValueError(f"Task coverage mismatch for {name}")
    scores = {}
    for name, rows in variants.items():
        successes = sum(float(row["task_reward"]) > 0 for row in rows.values())
        scores[name] = {"successes": successes, "samples": len(rows), "success_rate": successes / len(rows)}
    differences = []
    for task in sorted(expected):
        left = variants["foldagent"][task]["model_contexts"]
        right = variants["equivalent"][task]["model_contexts"]
        # Request snapshots exclude random request IDs. Different generated
        # outputs can cause later divergence even with equivalent executors.
        first = next((i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]), None)
        branches_match = variants["foldagent"][task].get("branch_model_contexts") == variants["equivalent"][task].get("branch_model_contexts")
        if first is not None or len(left) != len(right) or not branches_match:
            differences.append({"task_id": task, "first_different_request": first, "request_counts": [len(left), len(right)], "branches_match": branches_match})
    result = {"commit": manifest["commit"], "scores": scores, "equivalence_request_differences": differences,
              "note": "Paired diagnostics, not proof of a causal graph benefit. Request divergence requires inspection of preceding outputs."}
    (root / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["equivalence_request_differences"] else 0)
