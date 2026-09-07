#!/usr/bin/env python3
"""Build deterministic Search-R1 train/eval subsets for RL diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SEARCH_BENCHMARKS = (
    "searchR1_nq",
    "searchR1_triviaqa",
    "searchR1_popqa",
    "searchR1_hotpotqa",
    "searchR1_2wikimultihopqa",
    "searchR1_musique",
    "searchR1_bamboogle",
)


def stratified_sample(
    frame: pd.DataFrame,
    sources: tuple[str, ...],
    per_source: int,
    seed: int,
) -> pd.DataFrame:
    """Sample exactly ``per_source`` rows per source and deterministically mix them."""
    if "data_source" not in frame.columns:
        raise ValueError("input parquet is missing the data_source column")
    if per_source <= 0:
        raise ValueError("per_source must be positive")

    chunks = []
    counts = frame["data_source"].value_counts().to_dict()
    for offset, source in enumerate(sources):
        available = int(counts.get(source, 0))
        if available < per_source:
            raise ValueError(
                f"{source} has {available} rows, fewer than requested {per_source}"
            )
        chunk = frame.loc[frame["data_source"] == source].sample(
            n=per_source,
            random_state=seed + offset,
            replace=False,
        )
        chunks.append(chunk)

    sampled = pd.concat(chunks, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-source", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        default=",".join(SEARCH_BENCHMARKS),
        help="Comma-separated data_source values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = tuple(source.strip() for source in args.sources.split(",") if source.strip())
    if not sources:
        raise SystemExit("--sources must contain at least one data source")

    frame = pd.read_parquet(args.input)
    sampled = stratified_sample(frame, sources, args.per_source, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(args.output, index=False)

    counts = sampled["data_source"].value_counts().sort_index()
    print(f"wrote {len(sampled)} rows to {args.output}")
    for source, count in counts.items():
        print(f"  {source}: {count}")


if __name__ == "__main__":
    main()
