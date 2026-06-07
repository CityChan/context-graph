"""Load ScienceAgentBench tasks from CSV and assemble per-task dicts that
match ScienceAgentEnv.init_env's `extra_info` expectations.

Upstream CSV schema (from osunlp/ScienceAgentBench HF dataset):
    instance_id, domain, subtask_categories, github_name, task_inst,
    domain_knowledge, dataset_folder_tree, dataset_preview,
    src_file_or_path, gold_program_name, output_fname, eval_script_name

The full benchmark (input data files + gold programs) lives in a
SharePoint zip behind a password and is NOT shipped with the HF CSV.
Until that zip is unpacked into `benchmark_dir`, tasks load with empty
input_files (the env will still build a workdir, the agent will just
fail at the read step — useful for testing the harness path).
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Optional


def _build_instruction(row: dict) -> str:
    """Compose the user-facing instruction from CSV columns.

    The order mirrors the upstream baselines so our results stay comparable.
    """
    parts = []
    parts.append(row.get('task_inst', '').strip())

    dk = (row.get('domain_knowledge') or '').strip()
    if dk:
        parts.append("# Domain Knowledge\n" + dk)

    tree = (row.get('dataset_folder_tree') or '').strip()
    if tree:
        parts.append("# Dataset Folder Tree\n" + tree)

    preview = (row.get('dataset_preview') or '').strip()
    if preview:
        parts.append("# Dataset Preview\n" + preview)

    return "\n\n".join(parts).strip()


def _build_input_files(row: dict, benchmark_dir: Optional[str]) -> list[str]:
    """Resolve absolute paths to input files referenced by this task.

    Strategy: parse the `dataset_folder_tree` string for `|-- ... |---- foo`
    leaf lines and join them against `benchmark_dir/datasets/`. If
    `benchmark_dir` is None (smoke / pre-download phase), return [].
    """
    if not benchmark_dir:
        return []

    datasets_dir = os.path.join(benchmark_dir, "datasets")
    if not os.path.isdir(datasets_dir):
        return []

    tree = row.get('dataset_folder_tree') or ''
    files: list[str] = []
    current_subdir: str | None = None
    for raw_line in tree.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith('|--'):
            continue
        name = line.lstrip('|').lstrip('-').strip()
        is_subdir = name.endswith('/')
        name = name.rstrip('/')
        # Dash count discriminates depth: '|--' = 2 dashes (top-level),
        # '|----' = 4 dashes (one level inside the previous subdir).
        depth_dashes = line.count('-')
        if is_subdir:
            current_subdir = name
            continue
        if depth_dashes >= 4 and current_subdir:
            rel = os.path.join(current_subdir, name)
        else:
            # Top-level leaf (no subdir prefix). Possible but uncommon
            # in real ScienceAgentBench data.
            rel = name
        abs_path = os.path.join(datasets_dir, rel)
        if os.path.exists(abs_path):
            files.append(abs_path)
    return files


def load_sab_tasks(
    csv_path: str,
    benchmark_dir: Optional[str] = None,
    instance_ids: Optional[Iterable[int]] = None,
    workflow: str = 'code',
) -> list[dict]:
    """Read ScienceAgentBench.csv -> list of `extra_info` dicts ready for
    verl DataProto wrapping.

    Args:
        csv_path: path to ScienceAgentBench.csv (downloaded from HF).
        benchmark_dir: optional path to the unpacked benchmark (with a
            `datasets/` subdir). If None, input_files will be [].
        instance_ids: optional whitelist of instance_id ints to load. If
            None, load all rows.
        workflow: 'code' / 'code_branch' / 'code_graph' — passed through
            to env.

    Returns:
        list[dict] where each dict has:
            task_id, instruction, input_files, expected_output, workflow,
            domain, eval_script_name, gold_program_name.

    The dict matches what ScienceAgentEnv.init_env reads from
    item.non_tensor_batch['extra_info'].
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    keep: set[int] | None = None
    if instance_ids is not None:
        keep = {int(i) for i in instance_ids}

    out: list[dict] = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                iid = int(row['instance_id'])
            except (KeyError, ValueError):
                continue
            if keep is not None and iid not in keep:
                continue

            instruction = _build_instruction(row)
            input_files = _build_input_files(row, benchmark_dir)
            expected = (row.get('output_fname') or '').strip() or None

            out.append({
                'task_id': str(iid),
                'instruction': instruction,
                'input_files': input_files,
                'expected_output': expected,
                'workflow': workflow,
                'domain': (row.get('domain') or '').strip(),
                'eval_script_name': (row.get('eval_script_name') or '').strip(),
                'gold_program_name': (row.get('gold_program_name') or '').strip(),
                'subtask_categories': (row.get('subtask_categories') or '').strip(),
                'github_name': (row.get('github_name') or '').strip(),
            })
    return out
