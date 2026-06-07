#!/usr/bin/env python3
"""Generate ScienceAgentBench parquet for zero-shot eval.

Reads the upstream osunlp/ScienceAgentBench CSV (HuggingFace) and wraps each
row into the verl `{prompt, ability, extra_info}` schema that our agent
loops consume.

ScienceAgentBench is eval-only (no train split), so we emit ONE parquet
that doubles as both train_files and val_files in the sbatch script
(verl wants both pointed somewhere; val_only=True ensures train is never
touched).

Three parquets are emitted, differing only in `extra_info.workflow`:
  data/sab_test_code.parquet         (workflow=code,        for ReAct)
  data/sab_test_code_branch.parquet  (workflow=code_branch, for FoldAgent)
  data/sab_test_code_graph.parquet   (workflow=code_graph,  for ContextGraph)

Usage:
  python scripts/make_sab_data.py \
      --csv data/ScienceAgentBench.csv \
      --benchmark-dir data/sab_benchmark \
      --out-dir data

  # subset:
  python scripts/make_sab_data.py --csv ... --instance-ids 1,2,3 ...
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Add repo root for `from envs.* import ...`
_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.scienceagent_loader import load_sab_tasks


def _to_row(task: dict) -> dict:
    """Convert a loader-emitted task dict into the verl row schema."""
    return {
        "prompt": [{"role": "user", "content": task['instruction']}],
        "ability": "ScienceAgentBench",
        "extra_info": {
            # Fields read by ScienceAgentEnv.init_env
            "task_id": task['task_id'],
            "instruction": task['instruction'],
            "input_files": task['input_files'],
            "expected_output": task['expected_output'],
            "workflow": task['workflow'],
            # Helpful metadata for analysis (logged via env.stats / wandb)
            "domain": task['domain'],
            "eval_script_name": task['eval_script_name'],
            "gold_program_name": task['gold_program_name'],
            "subtask_categories": task['subtask_categories'],
            "github_name": task['github_name'],
            # Used by various agent loops to retrieve the user-facing task
            # text (mirrors LocalSearch's extra_info['query'] usage)
            "query": task['instruction'],
            "problem_statement": task['instruction'],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="Path to ScienceAgentBench.csv (HF download)")
    ap.add_argument("--benchmark-dir", default=None,
                    help=("Path to the unpacked full benchmark, with a "
                          "`datasets/` subdir. If omitted, parquets are "
                          "still emitted but extra_info.input_files is "
                          "empty (agent will fail at read time — useful for "
                          "harness validation only)."))
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--instance-ids", default=None,
                    help="Comma-separated whitelist of instance_id ints.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    iids = None
    if args.instance_ids:
        iids = [int(x.strip()) for x in args.instance_ids.split(',') if x.strip()]

    for workflow, suffix in [
        ('code', 'code'),
        ('code_branch', 'code_branch'),
        ('code_graph', 'code_graph'),
    ]:
        tasks = load_sab_tasks(
            csv_path=args.csv,
            benchmark_dir=args.benchmark_dir,
            instance_ids=iids,
            workflow=workflow,
        )
        if not tasks:
            print(f"[WARN] {workflow}: 0 tasks loaded from {args.csv}")
            continue

        rows = [_to_row(t) for t in tasks]
        df = pd.DataFrame(rows)
        out_path = os.path.join(args.out_dir, f"sab_test_{suffix}.parquet")
        df.to_parquet(out_path, index=False)

        n_with_inputs = sum(1 for t in tasks if t['input_files'])
        print(
            f"[OK] {workflow}: {len(tasks)} tasks -> {out_path}  "
            f"(with input files: {n_with_inputs}/{len(tasks)})"
        )


if __name__ == "__main__":
    main()
