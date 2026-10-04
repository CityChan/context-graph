"""Prepare immutable public tasks and separate WideSearch grading references."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_discoverybench_qwen35 import digest, save

WIDESEARCH_REVISION = "6531a7e5b497d44c8912407e0cb3dc95bd98cc09"
WIDESEARCH_EVALUATOR = "9825ba7b140b71d81b364793f86dabe4cfed6749"
SCIENCEWORLD_VERSION = "1.2.3"


def verify_evaluator(path):
    path = Path(path)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip() != WIDESEARCH_EVALUATOR:
        raise ValueError("WideSearch evaluator revision mismatch")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True).strip():
        raise ValueError("WideSearch evaluator has modified tracked files")


def prepare(benchmark, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "tasks.json").exists():
        raise ValueError("Prepared data already exists; reuse it or choose a new directory")
    if benchmark == "scienceworld":
        from scripts.make_scienceworld_data import collect_variations
        version = importlib.metadata.version("scienceworld")
        if version != SCIENCEWORLD_VERSION:
            raise ValueError(f"Expected scienceworld=={SCIENCEWORLD_VERSION}, got {version}")
        tasks = collect_variations("test")
        source = {"benchmark": benchmark, "version": version, "split": "test", "simplification": ""}
    else:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download("ByteDance-Seed/WideSearch", repo_type="dataset",
                        revision=WIDESEARCH_REVISION, allow_patterns=["widesearch.jsonl", "widesearch_gold/*.csv"]))
        rows = [json.loads(line) for line in (snapshot / "widesearch.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
        if len(rows) != 200 or len({r["instance_id"] for r in rows}) != 200:
            raise ValueError("Expected the complete 200-query WideSearch dataset")
        tasks, references = [], {}
        for row in rows:
            identity = row["instance_id"]
            evaluation = row["evaluation"]
            if isinstance(evaluation, str):
                evaluation = json.loads(evaluation)
            csv = snapshot / "widesearch_gold" / f"{identity}.csv"
            references[identity] = {"evaluation": evaluation, "csv": csv.read_text(encoding="utf8")}
            tasks.append({"task_id": identity, "query": row["query"], "language": row["language"]})
        save(output / "references.json", references)
        upstream = output / "official-evaluator"
        if not upstream.exists():
            subprocess.run(["git", "clone", "https://github.com/ByteDance-Seed/WideSearch.git", str(upstream)], check=True)
            subprocess.run(["git", "checkout", "--detach", WIDESEARCH_EVALUATOR], cwd=upstream, check=True)
        verify_evaluator(upstream)
        source = {"benchmark": benchmark, "dataset": "ByteDance-Seed/WideSearch", "split": "full",
                  "revision": WIDESEARCH_REVISION, "evaluator_revision": WIDESEARCH_EVALUATOR,
                  "references_sha256": digest(output / "references.json")}
    tasks.sort(key=lambda t: t["task_id"])
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Empty or duplicate task list")
    save(output / "tasks.json", {"source": source, "tasks": tasks})
    print(f"BENCHMARK_DATA_READY benchmark={benchmark} tasks={len(tasks)} path={output / 'tasks.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["scienceworld", "widesearch"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.benchmark, args.output)
