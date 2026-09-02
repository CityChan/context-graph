#!/usr/bin/env python3
"""Create deterministic, query-disjoint MiroVerse RL smoke splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def query_hash(row: dict) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("query_hash"):
        return str(extra["query_hash"])
    raise ValueError("seed row is missing extra_info.query_hash")


def split_rows(rows: list[dict], train_samples: int, validation_samples: int, seed: int):
    if train_samples <= 0 or validation_samples <= 0:
        raise ValueError("train_samples and validation_samples must be positive")
    unique = {}
    for row in rows:
        unique.setdefault(query_hash(row), row)
    hashes = sorted(unique)
    random.Random(seed).shuffle(hashes)
    required = train_samples + validation_samples
    if len(hashes) < required:
        raise ValueError(f"need {required} unique queries, found {len(hashes)}")
    validation_hashes = set(hashes[:validation_samples])
    train_hashes = set(hashes[validation_samples:required])
    train = [unique[value] for value in hashes if value in train_hashes]
    validation = [unique[value] for value in hashes if value in validation_hashes]
    return train, validation


def main() -> None:
    args = parse_args()
    rows = pd.read_parquet(args.input).to_dict(orient="records")
    train, validation = split_rows(rows, args.train_samples, args.validation_samples, args.seed)
    train_output = Path(args.train_output)
    validation_output = Path(args.validation_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(train_output, index=False)
    pd.DataFrame(validation).to_parquet(validation_output, index=False)
    train_hashes = {query_hash(row) for row in train}
    validation_hashes = {query_hash(row) for row in validation}
    manifest = {
        "schema_version": "contextgraph.miroverse_rl_smoke_split.v1",
        "input": args.input,
        "train_output": str(train_output),
        "validation_output": str(validation_output),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "query_overlap": len(train_hashes & validation_hashes),
        "seed": args.seed,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
