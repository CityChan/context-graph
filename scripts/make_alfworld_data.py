#!/usr/bin/env python3
"""Generate ALFWorld task data for FoldAgent / ContextGraph training.

Scans $ALFWORLD_DATA/json_2.1.1/{train,valid_seen,valid_unseen}/ for
.tw-pddl game files and emits parquet files referencing them by path.
"""

import os
import json
import argparse
import random
from collections import Counter
from pathlib import Path
import pandas as pd


def scan_alfworld_games(alfworld_data_path, max_train=None, max_test=None, seed=42):
    """Scan ALFWorld data directory for .tw-pddl game files."""
    json_dir = Path(alfworld_data_path) / "json_2.1.1"
    if not json_dir.exists():
        raise FileNotFoundError(f"ALFWorld json_2.1.1 not found at {json_dir}")

    train_tasks = []
    test_tasks = []
    skipped = Counter()

    splits = {"train": train_tasks, "valid_seen": test_tasks, "valid_unseen": test_tasks}

    for split_name, task_list in splits.items():
        split_dir = json_dir / split_name
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found")
            continue

        for game_file in sorted(split_dir.rglob("game.tw-pddl")):
            game_path = str(game_file)
            if "movable" in game_path or "Sliced" in game_path:
                skipped["unsupported"] += 1
                continue

            traj_file = game_file.parent / "traj_data.json"
            if not traj_file.exists():
                skipped["missing_traj_data"] += 1
                continue

            try:
                with game_file.open("r", encoding="utf-8") as f:
                    game_data = json.load(f)
                with traj_file.open("r", encoding="utf-8") as f:
                    traj_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                skipped["invalid_metadata"] += 1
                continue

            if not game_data.get("solvable", False):
                skipped["unsolvable"] += 1
                continue

            relative = game_file.relative_to(split_dir)
            parts = relative.parts
            task_type = traj_data.get(
                "task_type",
                parts[0] if len(parts) >= 2 else "unknown",
            )
            trial_name = parts[1] if len(parts) >= 2 else game_file.stem

            task_list.append({
                "task_id": f"alfworld_{split_name}_{task_type}_{trial_name}",
                "task_type": task_type,
                "game_file": str(game_file.absolute()),
                "task_desc": f"ALFWorld task: {task_type.replace('_', ' ')}",
                "split": split_name,
                "answer": "success",
            })

    rng = random.Random(seed)
    rng.shuffle(train_tasks)
    rng.shuffle(test_tasks)

    if max_train:
        train_tasks = train_tasks[:max_train]
    if max_test:
        test_tasks = test_tasks[:max_test]

    print(
        f"ALFWorld: {len(train_tasks)} train, {len(test_tasks)} test "
        f"(official filters skipped={dict(skipped)})"
    )
    return train_tasks, test_tasks


def to_row(task, workflow, ability):
    return {
        "prompt": [{"role": "user", "content": task['task_desc']}],
        "ability": ability,
        "extra_info": {
            "query": task["task_desc"],
            "answer": task.get("answer", "success"),
            "problem_statement": task["task_desc"],
            "task_desc": task["task_desc"],
            "task_type": task.get("task_type", ""),
            "game_file": task.get("game_file", ""),
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["real", "hard", "both"], default="both",
                        help="Which ability mode(s) to write. Default 'both' writes "
                             "real and hard variants in one go so you can switch by "
                             "filename later without regenerating.")
    parser.add_argument("--hard", action="store_true",
                        help="DEPRECATED alias for --mode=hard. Kept for back-compat.")
    parser.add_argument("--n_train", type=int, default=300)
    parser.add_argument("--n_val", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--legacy", choices=["real", "hard"], default="real",
                        help="Which mode the un-suffixed legacy filenames "
                             "(alfworld_train.parquet etc) should point to. "
                             "Default 'real'. Older 4-node 16h scripts read these.")
    parser.add_argument("--alfworld_data", default=None,
                        help="Path to ALFWorld data (default: ALFWORLD_DATA env or ~/.cache/alfworld)")
    args = parser.parse_args()

    # Back-compat: --hard implies --mode=hard
    if args.hard:
        args.mode = "hard"
    modes = ["real", "hard"] if args.mode == "both" else [args.mode]

    os.makedirs(args.out_dir, exist_ok=True)

    alfworld_data = args.alfworld_data or os.environ.get("ALFWORLD_DATA")
    if not alfworld_data:
        for p in [os.path.expanduser("~/.cache/alfworld"),
                  "/home1/09281/chc_1996/.cache/alfworld"]:
            if os.path.exists(p):
                alfworld_data = p
                break
    if not alfworld_data or not os.path.exists(alfworld_data):
        raise FileNotFoundError("ALFWorld data not found. Run: alfworld-download")

    print(f"Using ALFWorld data from: {alfworld_data}")
    train_tasks, val_tasks = scan_alfworld_games(
        alfworld_data, max_train=args.n_train, max_test=args.n_val, seed=args.seed
    )

    type_counts = Counter(t.get("task_type", "?") for t in train_tasks)
    print(f"Type distribution: {dict(type_counts)}")

    for mode in modes:
        ability = f"ALFWorld@{mode}"
        is_legacy = (mode == args.legacy)

        # FoldAgent version
        train_rows = [to_row(t, "alfworld", ability) for t in train_tasks]
        val_rows = [to_row(t, "alfworld", ability) for t in val_tasks]
        pd.DataFrame(train_rows).to_parquet(
            f"{args.out_dir}/alfworld_{mode}_train.parquet", index=False)
        pd.DataFrame(val_rows).to_parquet(
            f"{args.out_dir}/alfworld_{mode}_test.parquet", index=False)
        print(f"Wrote alfworld_{mode}_{{train,test}}.parquet "
              f"({len(train_rows)} / {len(val_rows)} rows, ability={ability})")

        # ContextGraph version
        train_rows_g = [to_row(t, "alfworld_graph", ability) for t in train_tasks]
        val_rows_g = [to_row(t, "alfworld_graph", ability) for t in val_tasks]
        pd.DataFrame(train_rows_g).to_parquet(
            f"{args.out_dir}/alfworld_graph_{mode}_train.parquet", index=False)
        pd.DataFrame(val_rows_g).to_parquet(
            f"{args.out_dir}/alfworld_graph_{mode}_test.parquet", index=False)
        print(f"Wrote alfworld_graph_{mode}_{{train,test}}.parquet "
              f"({len(train_rows_g)} / {len(val_rows_g)} rows, ability={ability})")

        if is_legacy:
            # Also emit the un-suffixed legacy filenames so older 4-node 16h
            # scripts (which read alfworld_train.parquet etc) keep working.
            pd.DataFrame(train_rows).to_parquet(
                f"{args.out_dir}/alfworld_train.parquet", index=False)
            pd.DataFrame(val_rows).to_parquet(
                f"{args.out_dir}/alfworld_test.parquet", index=False)
            pd.DataFrame(train_rows_g).to_parquet(
                f"{args.out_dir}/alfworld_graph_train.parquet", index=False)
            pd.DataFrame(val_rows_g).to_parquet(
                f"{args.out_dir}/alfworld_graph_test.parquet", index=False)
            print(f"  + legacy un-suffixed copies (ability={ability})")

    print("\nSample tasks:")
    for t in train_tasks[:5]:
        gf = t.get('game_file', '')
        gf_short = f" ({os.path.basename(os.path.dirname(gf))})" if gf else ""
        print(f"  [{t.get('task_type', '?')}] {t['task_desc']}{gf_short}")


if __name__ == "__main__":
    main()
