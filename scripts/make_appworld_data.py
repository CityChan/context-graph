#!/usr/bin/env python3
"""Build train-only ContextGraph parquets from the official AppWorld split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def collect_train_tasks(seed: int = 42) -> list[dict]:
    from appworld import load_task_ids

    task_ids = list(load_task_ids("train"))
    random.Random(seed).shuffle(task_ids)
    return [{"task_id": str(task_id), "split": "train"} for task_id in task_ids]


def to_row(task: dict, workflow: str) -> dict:
    label = f"AppWorld train task {task['task_id']}"
    return {
        "prompt": [{"role": "user", "content": label}],
        "ability": "AppWorld@train",
        "extra_info": {
            **task,
            "query": label,
            "problem_statement": label,
            "workflow": workflow,
            "answer": "success",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()

    train = collect_train_tasks(args.seed)
    if args.count_only:
        print(len(train))
        return
    if args.n_train > 0:
        train = train[: args.n_train]

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for workflow in ("appworld", "appworld_branch", "appworld_graph"):
        suffix = workflow.removeprefix("appworld").lstrip("_")
        prefix = "appworld" if not suffix else f"appworld_{suffix}"
        pd.DataFrame([to_row(task, workflow) for task in train]).to_parquet(
            output / f"{prefix}_train.parquet", index=False
        )
    print(f"AppWorld: wrote {len(train)} official train tasks to {output}")


if __name__ == "__main__":
    main()
