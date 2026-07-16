#!/usr/bin/env python3
"""Build GAIA parquets for ContextGraph/FoldAgent/ReAct search agents.

GAIA is a gated HuggingFace dataset. Run `huggingface-cli login` before this
script if the dataset is not already cached locally.

This first integration targets text-only GAIA items. Rows with attached files
are skipped by default because the current search agents do not yet expose
file, image, spreadsheet, or OCR tools.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

import pandas as pd


WORKFLOWS = [
    ("search", ""),
    ("search_branch", "_branch"),
    ("search_graph", "_graph"),
]


def _first(row: dict[str, Any], *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    lower = {str(k).lower().replace(" ", "_"): k for k in row.keys()}
    for name in names:
        key = lower.get(name.lower().replace(" ", "_"))
        if key is not None and row[key] is not None:
            return row[key]
    return default


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def _has_attachment(row: dict[str, Any]) -> bool:
    for name in ("file_name", "file_path", "file", "files", "attachment", "attachments"):
        if _nonempty(_first(row, name, default=None)):
            return True
    return False


def _level(row: dict[str, Any]) -> str:
    raw = _first(row, "Level", "level", default="")
    if isinstance(raw, (int, float)) and not pd.isna(raw):
        return str(int(raw))
    text = str(raw).strip()
    return text


def _task_id(row: dict[str, Any], idx: int) -> str:
    raw = _first(row, "task_id", "id", "instance_id", "Question ID", default="")
    return str(raw).strip() or f"gaia_{idx:05d}"


def _answer(row: dict[str, Any]) -> str:
    raw = _first(row, "Final answer", "final_answer", "answer", "Answer", default="")
    return str(raw).strip()


def _question(row: dict[str, Any]) -> str:
    raw = _first(row, "Question", "question", "query", "problem_statement", default="")
    return str(raw).strip()


def _row_to_task(row: dict[str, Any], idx: int, workflow: str) -> dict[str, Any] | None:
    question = _question(row)
    answer = _answer(row)
    if not question:
        return None
    task_id = _task_id(row, idx)
    level = _level(row)
    file_name = _first(row, "file_name", "file", default="")
    file_path = _first(row, "file_path", default="")
    extra = {
        "task_id": task_id,
        "instance_id": task_id,
        "query": question,
        "problem_statement": question,
        "answer": answer,
        "workflow": workflow,
        "level": level,
        "file_name": str(file_name or ""),
        "file_path": str(file_path or ""),
        "source": "gaia-benchmark/GAIA",
    }
    return {
        "prompt": [{"role": "user", "content": question}],
        "ability": "GAIA",
        "data_source": "gaia",
        "extra_info": extra,
        "reward_model": {"style": "rule", "ground_truth": answer},
    }


def _load_dataset(args):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency `datasets`. Install it or use --input-jsonl/--input-csv.") from exc

    kwargs = {"split": args.split, "trust_remote_code": True}
    if args.config:
        ds = load_dataset(args.dataset, args.config, **kwargs)
    else:
        ds = load_dataset(args.dataset, **kwargs)
    return [dict(row) for row in ds]


def _load_local(args):
    if args.input_jsonl:
        return pd.read_json(args.input_jsonl, lines=True).to_dict("records")
    if args.input_csv:
        return pd.read_csv(args.input_csv).to_dict("records")
    return None


def _filter_rows(rows: list[dict[str, Any]], args) -> tuple[list[dict[str, Any]], Counter]:
    counts: Counter = Counter()
    levels = {x.strip() for x in args.levels.split(",") if x.strip()} if args.levels else None
    kept: list[dict[str, Any]] = []
    for row in rows:
        counts["input"] += 1
        level = _level(row)
        if levels and level not in levels:
            counts["skip_level"] += 1
            continue
        if _has_attachment(row) and not args.include_files:
            counts["skip_attachment"] += 1
            continue
        if not _answer(row) and not args.allow_unlabeled:
            counts["skip_unlabeled"] += 1
            continue
        if not _question(row):
            counts["skip_no_question"] += 1
            continue
        kept.append(row)
        counts[f"level_{level or 'unknown'}"] += 1
        if args.max_samples > 0 and len(kept) >= args.max_samples:
            break
    counts["kept"] = len(kept)
    return kept, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gaia-benchmark/GAIA")
    parser.add_argument("--config", default="2023_all")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--input-jsonl", default=None,
                        help="Optional local JSONL with GAIA-like columns.")
    parser.add_argument("--input-csv", default=None,
                        help="Optional local CSV with GAIA-like columns.")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--levels", default=None,
                        help="Comma-separated level filter, e.g. 1,2. Default keeps all levels.")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--include-files", action="store_true",
                        help="Keep rows with file attachments. Current agents only see file metadata.")
    parser.add_argument("--allow-unlabeled", action="store_true",
                        help="Keep rows without gold answers, useful only for leaderboard submission generation.")
    args = parser.parse_args()

    rows = _load_local(args)
    if rows is None:
        rows = _load_dataset(args)

    rows, counts = _filter_rows(rows, args)
    if not rows:
        raise SystemExit(f"No GAIA rows kept. Counts: {dict(counts)}")

    os.makedirs(args.out_dir, exist_ok=True)
    split_name = args.split.replace("/", "_")
    prefix = args.out_prefix or f"gaia_{split_name}"

    for workflow, suffix in WORKFLOWS:
        tasks = []
        for idx, row in enumerate(rows):
            task = _row_to_task(row, idx, workflow)
            if task is not None:
                tasks.append(task)
        out_path = os.path.join(args.out_dir, f"{prefix}{suffix}.parquet")
        pd.DataFrame(tasks).to_parquet(out_path, index=False)
        print(f"[OK] {workflow}: {len(tasks)} rows -> {out_path}")

    print(f"Counts: {dict(counts)}")


if __name__ == "__main__":
    main()
