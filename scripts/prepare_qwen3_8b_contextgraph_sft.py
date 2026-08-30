#!/usr/bin/env python3
"""Assemble and audit the formal Qwen3-8B ContextGraph SFT dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "messages",
    "tools",
    "enable_thinking",
    "trajectory_id",
    "task_id",
    "domain",
    "task_reward",
    "graph_structural_ops",
    "graph_invalid_ops",
    "graph_trace_json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-prefix", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-length", type=int, default=8192)
    return parser.parse_args()


def collect_source_files(data_root: Path, run_prefixes: list[str]) -> list[Path]:
    files: list[Path] = []
    for run_prefix in run_prefixes:
        matches = sorted(data_root.glob(f"{run_prefix}_*/*/contextgraph_sft_*.parquet"))
        matches = [path for path in matches if path.name in {"contextgraph_sft_train.parquet", "contextgraph_sft_validation.parquet"}]
        if not matches:
            raise ValueError(f"no SFT parquet files found for run prefix {run_prefix!r} under {data_root}")
        files.extend(matches)
    return files


def validate_rows(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"SFT data is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("SFT data is empty")
    failures = {
        "task_reward": int((frame["task_reward"].astype(float) < 1.0).sum()),
        "structural_ops": int((frame["graph_structural_ops"].astype(int) < 1).sum()),
        "invalid_ops": int((frame["graph_invalid_ops"].astype(int) != 0).sum()),
        "missing_trace": int(frame["graph_trace_json"].isna().sum()),
        "missing_trajectory_id": int(frame["trajectory_id"].isna().sum()),
    }
    failures = {key: value for key, value in failures.items() if value}
    if failures:
        raise ValueError(f"formal SFT quality gates failed: {failures}")


def assemble_dataset(
    source_files: list[Path], validation_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    frames = []
    source_rows: dict[str, int] = {}
    for path in source_files:
        frame = pd.read_parquet(path)
        source_rows[str(path)] = len(frame)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    validate_rows(combined)
    rows_before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["trajectory_id"], keep="first").reset_index(drop=True)

    validation_keys: set[tuple[str, str]] = set()
    for domain, domain_frame in combined.groupby("domain", sort=True):
        task_ids = sorted(domain_frame["task_id"].astype(str).unique())
        random.Random(f"{seed}:{domain}").shuffle(task_ids)
        validation_count = max(1, round(len(task_ids) * validation_fraction))
        validation_count = min(validation_count, len(task_ids) - 1)
        validation_keys.update((str(domain), task_id) for task_id in task_ids[:validation_count])

    row_keys = list(zip(combined["domain"].astype(str), combined["task_id"].astype(str)))
    validation_selector = pd.Series([key in validation_keys for key in row_keys], index=combined.index)
    validation = combined.loc[validation_selector].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    training = combined.loc[~validation_selector].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_keys = set(zip(training["domain"].astype(str), training["task_id"].astype(str)))
    val_keys = set(zip(validation["domain"].astype(str), validation["task_id"].astype(str)))
    overlap = train_keys.intersection(val_keys)
    if overlap:
        raise ValueError(f"task leakage between train and validation: {sorted(overlap)[:5]}")
    summary = {
        "source_files": source_rows,
        "rows_before_dedup": rows_before_dedup,
        "duplicate_trajectories_removed": rows_before_dedup - len(combined),
        "train_rows": len(training),
        "validation_rows": len(validation),
        "train_tasks": len(train_keys),
        "validation_tasks": len(val_keys),
        "domains": sorted(combined["domain"].astype(str).unique()),
        "seed": seed,
        "validation_fraction": validation_fraction,
    }
    return training, validation, summary


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    return value


def audit_token_lengths(
    frame: pd.DataFrame, tokenizer_path: str, max_length: int
) -> tuple[dict[str, Any], list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, local_files_only=True)
    lengths = []
    for row in frame.itertuples(index=False):
        messages = to_builtin(row.messages)
        tools = to_builtin(row.tools)
        token_ids = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            enable_thinking=bool(row.enable_thinking),
            add_generation_prompt=False,
            tokenize=True,
        )
        lengths.append(len(token_ids))
    series = pd.Series(lengths, dtype="int64")
    thresholds = {}
    for threshold in (8192, 12288, 16384, 20480):
        over_threshold = int((series > threshold).sum())
        thresholds[str(threshold)] = {
            "over": over_threshold,
            "rate": float(over_threshold / len(series)),
        }
    return {
        "samples": len(lengths),
        "min": int(series.min()),
        "p50": int(series.quantile(0.50)),
        "p90": int(series.quantile(0.90)),
        "p95": int(series.quantile(0.95)),
        "p99": int(series.quantile(0.99)),
        "max": int(series.max()),
        "max_length": max_length,
        "over_max_length": int((series > max_length).sum()),
        "over_max_length_rate": float((series > max_length).mean()),
        "thresholds": thresholds,
    }, lengths


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    source_files = collect_source_files(data_root, args.run_prefix)
    training, validation, summary = assemble_dataset(source_files, args.validation_fraction, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.tokenizer:
        train_token_stats, train_lengths = audit_token_lengths(training, args.tokenizer, args.max_length)
        validation_token_stats, _ = audit_token_lengths(validation, args.tokenizer, args.max_length)
        summary["train_token_lengths"] = train_token_stats
        summary["validation_token_lengths"] = validation_token_stats
        longest_indices = sorted(range(len(train_lengths)), key=train_lengths.__getitem__, reverse=True)[:4]
        longest_output = output_dir / "contextgraph_sft_longest4.parquet"
        training.iloc[longest_indices].to_parquet(longest_output, index=False)
        summary["longest_smoke_output"] = str(longest_output)
        summary["longest_smoke_lengths"] = [train_lengths[index] for index in longest_indices]

    train_output = output_dir / "contextgraph_sft_train.parquet"
    validation_output = output_dir / "contextgraph_sft_validation.parquet"
    manifest_output = output_dir / "manifest.json"
    training.to_parquet(train_output, index=False)
    validation.to_parquet(validation_output, index=False)
    summary.update({"train_output": str(train_output), "validation_output": str(validation_output)})
    manifest_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
