"""Validate the three-shard BC-P/local-GAIA evaluator's saved evidence."""

import json
from pathlib import Path


def read_evaluation(root):
    root = Path(root)
    manifests = [json.loads((root / f"manifest-{rank}.json").read_text(encoding="utf-8")) for rank in range(3)]
    first = manifests[0]
    fields = ("source_sha256", "indices", "commit", "model_path", "method", "config", "seed", "judge_model")
    indices = first["indices"]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("Empty or duplicated selection in manifest")
    results = []
    for rank, manifest in enumerate(manifests):
        if manifest.get("benchmark", "bcp") != first.get("benchmark", "bcp"):
            raise ValueError("Shard benchmark mismatch")
        if manifest.get("model") != first.get("model"):
            raise ValueError("Shard model mismatch")
        if any(manifest[key] != first[key] for key in fields):
            raise ValueError("Shard provenance mismatch")
        if manifest.get("retrieval", "local_bcp_corpus") != first.get("retrieval", "local_bcp_corpus"):
            raise ValueError("Shard retrieval mismatch")
        if manifest.get("rank", rank) != rank:
            raise ValueError("Shard rank mismatch")
        rows = [json.loads(line) for line in (root / f"results-{rank}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if sorted(row["source_index"] for row in rows) != sorted(indices[rank::3]):
            raise ValueError("Missing, duplicated, or misplaced evaluation rows")
        for row in rows:
            if row["task_reward"] not in (0, 1) or not isinstance(row["is_finish"], bool):
                raise ValueError("Nonbinary task reward or invalid finish flag")
        results.extend(rows)
    summary = {
        "method": first["method"], "benchmark": first.get("benchmark", "bcp"),
        "count": len(results),
        "task_successes": sum(row["task_reward"] for row in results),
        "task_accuracy": sum(row["task_reward"] for row in results) / len(results),
        "finished": sum(row["is_finish"] for row in results),
        "execution_errors": sum(row["status"] != "ok" for row in results),
        "judge_parse_failures": sum(row.get("env_stats", {}).get("judge_parse_failure", 0) for row in results),
    }
    return manifests, sorted(results, key=lambda row: row["source_index"]), summary
