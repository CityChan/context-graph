#!/usr/bin/env python3
"""Create matched train/holdout parquets from GAIA's public validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import pandas as pd


VARIANTS = ("", "_branch", "_graph")


def _extra_info(value) -> Mapping:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"extra_info must be a mapping, got {type(value).__name__}")
    return value


def _identity(row: pd.Series) -> tuple[str, str]:
    extra = _extra_info(row["extra_info"])
    task_id = str(extra.get("task_id") or extra.get("instance_id") or "").strip()
    if not task_id:
        raise ValueError("GAIA row is missing task_id/instance_id in extra_info")
    return task_id, str(extra.get("level") or "unknown")


def select_holdout_ids(records: list[tuple[str, str]], fraction: float, seed: int) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be between 0 and 1")
    groups: dict[str, list[str]] = defaultdict(list)
    for task_id, level in records:
        groups[level].append(task_id)

    holdout: set[str] = set()
    for level, task_ids in sorted(groups.items()):
        ordered = sorted(
            task_ids,
            key=lambda task_id: hashlib.sha256(f"{seed}:{level}:{task_id}".encode()).hexdigest(),
        )
        count = min(len(ordered) - 1, max(1, round(len(ordered) * fraction))) if len(ordered) > 1 else 0
        holdout.update(ordered[:count])
    if not holdout:
        raise ValueError("holdout split is empty")
    return holdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    frames = {suffix: pd.read_parquet(input_dir / f"gaia_validation{suffix}.parquet") for suffix in VARIANTS}

    base_records = [_identity(row) for _, row in frames[""].iterrows()]
    base_ids = [task_id for task_id, _ in base_records]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("duplicate task IDs in GAIA validation parquet")
    holdout_ids = select_holdout_ids(base_records, args.holdout_fraction, args.seed)
    train_ids = set(base_ids) - holdout_ids

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, frame in frames.items():
        ids = frame.apply(lambda row: _identity(row)[0], axis=1)
        if set(ids) != set(base_ids):
            raise ValueError(f"GAIA workflow parquet {suffix or 'baseline'} has mismatched task IDs")
        train_frame = frame[ids.isin(train_ids)].copy()
        holdout_frame = frame[ids.isin(holdout_ids)].copy()
        train_path = output_dir / f"gaia_train{suffix}.parquet"
        holdout_path = output_dir / f"gaia_holdout{suffix}.parquet"
        train_frame.to_parquet(train_path, index=False)
        holdout_frame.to_parquet(holdout_path, index=False)
        print(f"[OK] {suffix or 'baseline'}: train={len(train_frame)} holdout={len(holdout_frame)}")


if __name__ == "__main__":
    main()
