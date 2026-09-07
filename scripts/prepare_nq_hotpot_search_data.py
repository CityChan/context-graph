#!/usr/bin/env python3
"""Convert Search-R1 NQ/HotpotQA parquet files for ContextGraph LocalSearch."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SOURCE_NAMES = {
    "nq": "searchR1_nq",
    "searchr1_nq": "searchR1_nq",
    "hotpotqa": "searchR1_hotpotqa",
    "searchr1_hotpotqa": "searchR1_hotpotqa",
}


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value.get("target", value.get("answers", []))
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    result = []
    for item in values:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def answer_aliases(row: pd.Series) -> list[str]:
    aliases = _as_strings(row.get("golden_answers"))
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        for answer in _as_strings(reward_model.get("ground_truth")):
            if answer not in aliases:
                aliases.append(answer)
    return aliases


def convert_frame(frame: pd.DataFrame, split: str, seed: int) -> tuple[pd.DataFrame, Counter]:
    missing = {"question", "data_source"}.difference(frame.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")

    rows = []
    counts: Counter = Counter()
    for source_index, row in frame.iterrows():
        source_raw = str(row.get("data_source", "")).strip()
        source = SOURCE_NAMES.get(source_raw.lower())
        if source is None:
            counts[f"skipped:{source_raw or 'empty'}"] += 1
            continue
        question = str(row.get("question", "")).strip()
        aliases = answer_aliases(row)
        if not question:
            counts["rejected:empty_question"] += 1
            continue
        if not aliases:
            counts["rejected:missing_answer"] += 1
            continue
        source_id = str(row.get("id", source_index)).strip()
        digest = hashlib.sha256(f"{source}:{source_id}:{question}".encode("utf-8")).hexdigest()[:16]
        task_id = f"{source}_{split}_{digest}"
        extra_info = {
            "task_id": task_id,
            "instance_id": task_id,
            "query": question,
            "problem_statement": question,
            "answer": aliases[0],
            "answer_aliases": aliases,
            "reward_mode": "searchr1_em",
            "workflow": "searchr1",
            "split": split,
            "source": "PeterJinGo/nq_hotpotqa_train",
            "source_index": int(source_index),
        }
        rows.append(
            {
                "prompt": [{"role": "user", "content": question}],
                "ability": "LocalSearch",
                "data_source": source,
                "extra_info": extra_info,
                "reward_model": {"style": "rule", "ground_truth": aliases[0]},
            }
        )
        counts[source] += 1

    if not rows:
        raise ValueError(f"no NQ/HotpotQA rows accepted for {split}: {dict(counts)}")
    converted = pd.DataFrame(rows)
    converted = converted.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return converted, counts


def balanced_sample(frame: pd.DataFrame, per_source: int, seed: int) -> pd.DataFrame:
    if per_source <= 0:
        raise ValueError("per_source must be positive")
    chunks = []
    for offset, source in enumerate(("searchR1_nq", "searchR1_hotpotqa")):
        source_frame = frame.loc[frame["data_source"] == source]
        if len(source_frame) < per_source:
            raise ValueError(f"{source} has {len(source_frame)} rows; requested {per_source}")
        chunks.append(source_frame.sample(n=per_source, random_state=seed + offset))
    return pd.concat(chunks, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", required=True, type=Path)
    parser.add_argument("--validation-input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-per-source", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, train_counts = convert_frame(pd.read_parquet(args.train_input), "train", args.seed)
    validation, validation_counts = convert_frame(
        pd.read_parquet(args.validation_input), "validation", args.seed + 1
    )
    validation_diag = balanced_sample(validation, args.validation_per_source, args.seed + 2)

    outputs = {
        "train": args.output_dir / "train.parquet",
        "validation": args.output_dir / "validation.parquet",
        "validation_diag": args.output_dir / "validation_diag.parquet",
    }
    train.to_parquet(outputs["train"], index=False)
    validation.to_parquet(outputs["validation"], index=False)
    validation_diag.to_parquet(outputs["validation_diag"], index=False)

    manifest = {
        "schema_version": "contextgraph.searchr1_nq_hotpotqa.v1",
        "source": "PeterJinGo/nq_hotpotqa_train",
        "seed": args.seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "validation_diag_rows": len(validation_diag),
        "train_counts": dict(train_counts),
        "validation_counts": dict(validation_counts),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
