#!/usr/bin/env python3
"""Generate ALFWorld task data for FoldAgent / ContextGraph training.

Two modes:
  --mode mock   : Synthetic tasks with mock state machine (no alfworld needed)
  --mode real   : Real ALFWorld game files from alfworld-download

Real mode scans $ALFWORLD_DATA/json_2.1.1/{train,valid_seen,valid_unseen}/
for .tw-pddl game files and creates parquet with game_file paths.
"""

import os
import argparse
import random
from pathlib import Path
import pandas as pd


# ── Mock task generation (same as before) ──

OBJECTS = [
    "apple", "tomato", "potato", "lettuce", "bread", "egg", "mug",
    "cup", "plate", "bowl", "knife", "fork", "spoon", "spatula",
    "pen", "pencil", "book", "newspaper", "remote", "candle",
    "soap", "cloth", "sponge", "key", "watch", "phone",
]

MOCK_RECEPTACLES = {
    "kitchen": ["countertop 1", "countertop 2", "cabinet 1", "cabinet 2",
                "drawer 1", "fridge 1", "sink 1", "sinkbasin 1",
                "stoveburner 1", "microwave 1", "garbagecan 1"],
    "living room": ["coffeetable 1", "sofa 1", "shelf 1", "shelf 2",
                    "armchair 1", "sidetable 1"],
    "bedroom": ["bed 1", "dresser 1", "desk 1", "drawer 2",
                "sidetable 2", "shelf 3"],
    "bathroom": ["sinkbasin 2", "toilet 1", "bathtub 1",
                 "shelf 4", "cabinet 3"],
}

TASK_TYPES = [
    {"type": "pick_three_split_containers",
     "template": "Put a {obj1} on the {recep1}, put a {obj2} on the {recep2}, AND put a {obj3} on the {recep3}. The {obj1} and {obj2} are in {loc_a}; the {obj3} is in {loc_b}."},
]

# Containers suitable as shared storage (can hold multiple items)
SHARED_CONTAINERS = ["fridge 1", "cabinet 1", "cabinet 2", "drawer 1", "drawer 2",
                     "cabinet 3", "shelf 1", "shelf 2", "dresser 1"]


def generate_mock_tasks(n=300, seed=42):
    """Multi-object task: 3 objects split across 2 containers (2+1), 3 different receptacles."""
    random.seed(seed)
    all_receptacles = [r for room in MOCK_RECEPTACLES.values() for r in room]
    tasks = []
    for _ in range(n):
        task_type = random.choice(TASK_TYPES)
        obj1, obj2, obj3 = random.sample(OBJECTS, 3)
        loc_a, loc_b = random.sample(SHARED_CONTAINERS, 2)
        # 3 distinct target receptacles, none of which are the containers
        candidates = [r for r in all_receptacles if r not in (loc_a, loc_b)]
        recep1, recep2, recep3 = random.sample(candidates, 3)
        tasks.append({
            "task_desc": task_type["template"].format(
                obj1=obj1, obj2=obj2, obj3=obj3,
                recep1=recep1, recep2=recep2, recep3=recep3,
                loc_a=loc_a, loc_b=loc_b,
            ),
            "task_type": task_type["type"],
            "target_objects": [obj1, obj2, obj3],
            "target_receptacles": [recep1, recep2, recep3],
            "object_location_map": {obj1: loc_a, obj2: loc_a, obj3: loc_b},
            # keep single object_location for backward compat (used by env if map missing)
            "object_location": loc_a,
            "answer": "success",
        })
    return tasks


# ── Real ALFWorld game file scanning ──

def scan_alfworld_games(alfworld_data_path, max_train=None, max_test=None, seed=42):
    """Scan ALFWorld data directory for .tw-pddl game files."""
    json_dir = Path(alfworld_data_path) / "json_2.1.1"
    if not json_dir.exists():
        raise FileNotFoundError(f"ALFWorld json_2.1.1 not found at {json_dir}")

    train_tasks = []
    test_tasks = []

    splits = {"train": train_tasks, "valid_seen": test_tasks, "valid_unseen": test_tasks}

    for split_name, task_list in splits.items():
        split_dir = json_dir / split_name
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found")
            continue

        for game_file in sorted(split_dir.rglob("*.tw-pddl")):
            relative = game_file.relative_to(split_dir)
            parts = relative.parts
            task_type = parts[0] if len(parts) >= 2 else "unknown"
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

    print(f"Real ALFWorld: {len(train_tasks)} train, {len(test_tasks)} test")
    return train_tasks, test_tasks


def to_row(task, workflow, mode="mock"):
    ability = f"ALFWorld@{mode}"
    return {
        "prompt": [{"role": "user", "content": task['task_desc']}],
        "ability": ability,
        "extra_info": {
            "query": task["task_desc"],
            "answer": task.get("answer", "success"),
            "problem_statement": task["task_desc"],
            "task_desc": task["task_desc"],
            "task_type": task.get("task_type", ""),
            "target_objects": task.get("target_objects", []),
            "target_receptacles": task.get("target_receptacles", []),
            "object_location": task.get("object_location", ""),
            "game_file": task.get("game_file", ""),
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--hard", action="store_true",
                        help="Hard mode: hide admissible commands (MemexRL style). Ability becomes ALFWorld@<mode>_hard.")
    parser.add_argument("--n_train", type=int, default=300)
    parser.add_argument("--n_val", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--alfworld_data", default=None,
                        help="Path to ALFWorld data (default: ALFWORLD_DATA env or ~/.cache/alfworld)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "real":
        # Find ALFWorld data path
        alfworld_data = args.alfworld_data
        if not alfworld_data:
            alfworld_data = os.environ.get("ALFWORLD_DATA")
        if not alfworld_data:
            # Try default locations
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
        mode = "real"
    else:
        all_tasks = generate_mock_tasks(args.n_train + args.n_val, args.seed)
        train_tasks = all_tasks[:args.n_train]
        val_tasks = all_tasks[args.n_train:args.n_train + args.n_val]
        mode = "mock"
        print(f"Mock ALFWorld: {len(train_tasks)} train, {len(val_tasks)} val")

    # Count task types
    from collections import Counter
    type_counts = Counter(t.get("task_type", "?") for t in train_tasks)
    print(f"Type distribution: {dict(type_counts)}")

    # Ability string: ALFWorld@<mode>[_hard]
    ability_mode = f"{mode}_hard" if args.hard else mode

    # FoldAgent version
    train_rows = [to_row(t, "alfworld", ability_mode) for t in train_tasks]
    val_rows = [to_row(t, "alfworld", ability_mode) for t in val_tasks]
    pd.DataFrame(train_rows).to_parquet(f"{args.out_dir}/alfworld_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{args.out_dir}/alfworld_test.parquet", index=False)
    print(f"Wrote alfworld_train.parquet ({len(train_rows)} rows, ability=ALFWorld@{ability_mode})")

    # ContextGraph version
    train_rows_g = [to_row(t, "alfworld_graph", ability_mode) for t in train_tasks]
    val_rows_g = [to_row(t, "alfworld_graph", ability_mode) for t in val_tasks]
    pd.DataFrame(train_rows_g).to_parquet(f"{args.out_dir}/alfworld_graph_train.parquet", index=False)
    pd.DataFrame(val_rows_g).to_parquet(f"{args.out_dir}/alfworld_graph_test.parquet", index=False)
    print(f"Wrote alfworld_graph_train.parquet ({len(train_rows_g)} rows, ability=ALFWorld@{ability_mode})")

    # Samples
    print("\nSample tasks:")
    for t in train_tasks[:5]:
        gf = t.get('game_file', '')
        gf_short = f" ({os.path.basename(os.path.dirname(gf))})" if gf else ""
        print(f"  [{t.get('task_type', '?')}] {t['task_desc']}{gf_short}")


if __name__ == "__main__":
    main()
