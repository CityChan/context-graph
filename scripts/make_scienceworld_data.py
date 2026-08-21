#!/usr/bin/env python3
"""Build ContextGraph parquets from official ScienceWorld task splits."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


def collect_variations(split: str) -> list[dict]:
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv()
    tasks: list[dict] = []
    try:
        for task_name in env.get_task_names():
            env.load(taskName=task_name, variationIdx=0, simplificationStr="", generateGoldPath=False)
            if split == "train":
                variations = env.get_variations_train()
            elif split == "dev":
                variations = env.get_variations_dev()
            else:
                variations = env.get_variations_test()
            tasks.extend(
                {
                    "task_id": f"scienceworld_{split}_{task_name}_{variation_idx}",
                    "task_name": task_name,
                    "variation_idx": int(variation_idx),
                    "split": split,
                }
                for variation_idx in variations
            )
    finally:
        env.close()
    return tasks


def to_row(task: dict, workflow: str) -> dict:
    label = f"ScienceWorld task {task['task_name']} variation {task['variation_idx']}"
    return {
        "prompt": [{"role": "user", "content": label}],
        "ability": "ScienceWorld@real",
        "extra_info": {
            **task,
            "query": label,
            "problem_statement": label,
            "workflow": workflow,
            "simplification": "",
            "answer": "success",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-val", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train = collect_variations("train")
    dev = collect_variations("dev")
    rng.shuffle(train)
    rng.shuffle(dev)
    if args.n_train > 0:
        train = train[: args.n_train]
    if args.n_val > 0:
        dev = dev[: args.n_val]

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for workflow in ("scienceworld", "scienceworld_branch", "scienceworld_graph"):
        suffix = workflow.removeprefix("scienceworld").lstrip("_")
        prefix = "scienceworld" if not suffix else f"scienceworld_{suffix}"
        pd.DataFrame([to_row(task, workflow) for task in train]).to_parquet(
            output / f"{prefix}_train.parquet", index=False
        )
        pd.DataFrame([to_row(task, workflow) for task in dev]).to_parquet(
            output / f"{prefix}_val.parquet", index=False
        )
    print(f"ScienceWorld: wrote {len(train)} train and {len(dev)} dev tasks to {output}")


if __name__ == "__main__":
    main()
