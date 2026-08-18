#!/usr/bin/env python3
"""Build verl train/validation parquets from the official D3-Gym metadata.

The upstream Hugging Face release calls its only split ``train``.  For honest
in-domain validation we create a deterministic *repository-disjoint* holdout:
all tasks from one ``original_repo`` stay on the same side of the split.

Examples:

    python scripts/make_d3gym_data.py --out-dir data
    python scripts/make_d3gym_data.py --source data/d3gym_metadata.parquet --out-dir data
    python scripts/make_d3gym_data.py --source ... --task-ids task_1,task_2 --val-fraction 0.5
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.d3gym_env import extract_input_paths


WORKFLOWS = (
    ("code", "code"),
    ("code_branch", "code_branch"),
    ("code_graph", "code_graph"),
)


def extract_expected_outputs(eval_script: str) -> list[str]:
    """Best-effort static extraction of required ``pred_results`` files."""
    if not eval_script:
        return []
    values: set[str] = set()
    try:
        tree = ast.parse(eval_script)
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
    except SyntaxError:
        literals = re.findall(r"['\"]([^'\"]+)['\"]", eval_script)

    for value in literals:
        normalized = value.replace("\\", "/")
        matches = list(re.finditer(r"pred_results/([A-Za-z0-9_./+ -]+\.[A-Za-z0-9]{1,8})", normalized))
        for match in matches:
            values.add(match.group(1).rstrip(".,;: )]"))
        base = os.path.basename(normalized)
        if not matches and re.fullmatch(r"pred_[A-Za-z0-9_.+ -]+\.[A-Za-z0-9]{1,8}", base):
            values.add(base)
    return sorted(values)


def _load_source(source: str | None, dataset: str) -> pd.DataFrame:
    if not source:
        from datasets import load_dataset

        return load_dataset(dataset, split="train").to_pandas()
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"D3-Gym metadata source not found: {source}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported D3-Gym metadata format: {path.suffix}")


def _holdout_repo(repo: str, seed: int, val_fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{repo}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < val_fraction


def split_by_repository(df: pd.DataFrame, seed: int, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return repository-disjoint train and validation frames."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be strictly between 0 and 1")
    repos = df["original_repo"].fillna("").astype(str)
    if repos.nunique() < 2:
        raise ValueError("a repository-disjoint split requires tasks from at least two original repositories")
    is_val = repos.map(lambda repo: _holdout_repo(repo, seed, val_fraction))
    if not bool(is_val.any()) or bool(is_val.all()):
        # Tiny --task-ids subsets can hash entirely to one side.  Pick one
        # repository deterministically so smoke data remains usable.
        unique = sorted(set(repos))
        chosen = unique[seed % len(unique)]
        is_val = repos == chosen
    return df.loc[~is_val].reset_index(drop=True), df.loc[is_val].reset_index(drop=True)


def _truncate(value: str, max_chars: int) -> str:
    value = value or ""
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n... [{len(value) - max_chars} chars truncated]"


def _to_row(
    record: dict,
    *,
    workflow: str,
    runtime: str,
    image_template: str,
    local_task_root: str | None,
    preview_chars: int,
) -> dict:
    task_id = str(record["task_id"])
    task_instruction = str(record.get("task_instruction") or "").strip()
    preview = _truncate(str(record.get("dataset_previews") or "").strip(), preview_chars)
    instruction = task_instruction
    if preview:
        instruction += "\n\n# Dataset previews\n" + preview
    eval_script = str(record.get("eval_script") or "")
    task_dir = None
    if local_task_root:
        candidate = os.path.abspath(os.path.join(local_task_root, task_id))
        if os.path.isdir(candidate):
            task_dir = candidate
    extra_info = {
        "task_id": task_id,
        "instruction": instruction,
        "query": instruction,
        "problem_statement": instruction,
        "workflow": workflow,
        "image": image_template.format(task_id=task_id),
        "runtime": runtime,
        "task_dir": task_dir,
        "expected_outputs": extract_expected_outputs(eval_script),
        "input_paths": extract_input_paths(task_instruction),
        "eval_script": eval_script,
        "original_repo": str(record.get("original_repo") or ""),
        "discipline": str(record.get("discipline") or ""),
    }
    return {
        "prompt": [{"role": "user", "content": instruction}],
        "ability": "D3Gym",
        "extra_info": extra_info,
    }


def build_rows(
    records: Iterable[dict],
    *,
    workflow: str,
    runtime: str,
    image_template: str,
    local_task_root: str | None,
    preview_chars: int,
) -> list[dict]:
    return [
        _to_row(
            record,
            workflow=workflow,
            runtime=runtime,
            image_template=image_template,
            local_task_root=local_task_root,
            preview_chars=preview_chars,
        )
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Local parquet/csv/json metadata; otherwise load Hugging Face")
    parser.add_argument("--dataset", default="osunlp/D3-Gym")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--task-ids", help="Comma-separated task IDs for a smoke subset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--runtime", choices=("auto", "docker", "apptainer", "singularity", "local"), default="auto")
    parser.add_argument("--image-template", default="hananemoussa/d3-gym:{task_id}")
    parser.add_argument("--local-task-root")
    parser.add_argument("--preview-chars", type=int, default=6000)
    args = parser.parse_args()

    df = _load_source(args.source, args.dataset)
    required = {"task_id", "task_instruction", "original_repo"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"D3-Gym metadata is missing required columns: {missing}")
    if args.task_ids:
        keep = {value.strip() for value in args.task_ids.split(",") if value.strip()}
        df = df[df["task_id"].astype(str).isin(keep)].reset_index(drop=True)
    if df.empty:
        raise ValueError("no D3-Gym tasks selected")

    train, val = split_by_repository(df, args.seed, args.val_fraction)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "source": args.source,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_tasks": len(train),
        "val_tasks": len(val),
        "train_repositories": sorted(train["original_repo"].astype(str).unique().tolist()),
        "val_repositories": sorted(val["original_repo"].astype(str).unique().tolist()),
    }
    Path(args.out_dir, "d3gym_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    for workflow, suffix in WORKFLOWS:
        for split_name, split_df in (("train", train), ("val", val)):
            rows = build_rows(
                split_df.to_dict("records"),
                workflow=workflow,
                runtime=args.runtime,
                image_template=args.image_template,
                local_task_root=args.local_task_root,
                preview_chars=args.preview_chars,
            )
            out = os.path.join(args.out_dir, f"d3gym_{split_name}_{suffix}.parquet")
            pd.DataFrame(rows).to_parquet(out, index=False)
            print(f"[OK] {workflow} {split_name}: {len(rows)} tasks -> {out}")
    print(
        f"[OK] repository-disjoint split: train={len(train)} tasks/{train['original_repo'].nunique()} repos, "
        f"val={len(val)} tasks/{val['original_repo'].nunique()} repos"
    )


if __name__ == "__main__":
    main()
