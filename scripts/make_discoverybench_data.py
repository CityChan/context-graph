#!/usr/bin/env python3
"""Download/convert DiscoveryBench for zero-shot evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.discoverybench_loader import load_discoverybench_tasks


def _snapshot(download_dir: str | None) -> str:
    from huggingface_hub import snapshot_download
    kwargs = dict(
        repo_id="allenai/discoverybench",
        repo_type="dataset",
    )
    if download_dir:
        kwargs["local_dir"] = download_dir
    return snapshot_download(**kwargs)


def _row(task: dict) -> dict:
    # DiscoveryBench metadata is heterogeneous across tasks and occasionally
    # contains empty dictionaries (for example ``{"element": {}}``).  Arrow
    # cannot represent an empty struct in Parquet, so keep the complete
    # metadata as JSON and decode it in DiscoveryBenchEnv.
    parquet_task = {
        **task,
        "metadata": json.dumps(
            task.get("metadata", {}), ensure_ascii=False, sort_keys=True
        ),
    }
    return {
        "prompt": [{"role": "user", "content": task["instruction"]}],
        "ability": "DiscoveryBench",
        "extra_info": {
            **parquet_task,
            "problem_statement": task["instruction"],
            "workdir_root_env": "DISCOVERYBENCH_WORKDIR_ROOT",
            "workdir_prefix": "discoverybench_",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="data/discoverybench")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--dataset-type", choices=("real", "synth"), default="real")
    parser.add_argument("--split", default="test")
    parser.add_argument("--task-ids", default=None,
                        help="Comma-separated exact task IDs to retain")
    parser.add_argument("--include-domain-knowledge", action="store_true")
    parser.add_argument("--include-workflow-tags", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir
    if not args.no_download:
        benchmark_dir = _snapshot(benchmark_dir)
    elif not Path(benchmark_dir).is_dir():
        parser.error(f"--no-download specified but directory is missing: {benchmark_dir}")

    selected = None
    if args.task_ids:
        selected = {value.strip() for value in args.task_ids.split(",") if value.strip()}
    os.makedirs(args.out_dir, exist_ok=True)

    for workflow, suffix in (
        ("code", "code"),
        ("code_branch", "code_branch"),
        ("code_graph", "code_graph"),
    ):
        tasks = load_discoverybench_tasks(
            benchmark_dir=benchmark_dir,
            dataset_type=args.dataset_type,
            split=args.split,
            workflow=workflow,
            include_domain_knowledge=args.include_domain_knowledge,
            include_workflow_tags=args.include_workflow_tags,
            task_ids=selected,
        )
        output = Path(args.out_dir) / (
            f"discoverybench_{args.dataset_type}_{args.split}_{suffix}.parquet"
        )
        pd.DataFrame([_row(task) for task in tasks]).to_parquet(output, index=False)
        print(f"[OK] {workflow}: {len(tasks)} queries -> {output}")


if __name__ == "__main__":
    main()
