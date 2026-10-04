"""Prepare immutable public ScienceWorld test tasks."""
from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_discoverybench_qwen35 import save

SCIENCEWORLD_VERSION = "1.2.3"


def prepare(benchmark, output):
    if benchmark != "scienceworld":
        raise ValueError("Expected scienceworld")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "tasks.json").exists():
        raise ValueError("Prepared data already exists; reuse it or choose a new directory")
    from scripts.make_scienceworld_data import collect_variations
    version = importlib.metadata.version("scienceworld")
    if version != SCIENCEWORLD_VERSION:
        raise ValueError(f"Expected scienceworld=={SCIENCEWORLD_VERSION}, got {version}")
    tasks = collect_variations("test")
    source = {"benchmark": benchmark, "version": version, "split": "test", "simplification": ""}
    tasks.sort(key=lambda t: t["task_id"])
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Empty or duplicate task list")
    save(output / "tasks.json", {"source": source, "tasks": tasks})
    print(f"BENCHMARK_DATA_READY benchmark={benchmark} tasks={len(tasks)} path={output / 'tasks.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=["scienceworld"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.benchmark, args.output)
