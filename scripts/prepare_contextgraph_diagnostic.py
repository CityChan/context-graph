"""Freeze an outcome-independent BC-P subset for paired memory ablations."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path


def main():
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    data = pd.read_parquet(source)
    if args.samples != -1 and not 1 <= args.samples <= len(data):
        raise ValueError("samples must be -1 or between 1 and dataset size")
    indices = list(range(len(data))) if args.samples == -1 else sorted(random.Random(args.seed).sample(range(len(data)), args.samples))
    selected = data.iloc[indices].copy()
    metadata = []
    for index, original in zip(indices, selected["extra_info"]):
        value = dict(original or {})
        value["task_id"] = f"bcp-diagnostic-{index}"
        metadata.append(value)
    output.mkdir(parents=True, exist_ok=False)
    for name, workflow in (("foldagent", "search_branch"), ("graph", "search_graph")):
        variant = selected.copy()
        variant["extra_info"] = [dict(value, workflow=workflow) for value in metadata]
        variant.to_parquet(output / f"{name}.parquet", index=False)
    manifest = {
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "indices": indices, "seed": args.seed, "samples": len(indices),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "selection": "uniform without replacement; no outcomes used",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
