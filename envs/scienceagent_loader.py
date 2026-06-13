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
import re
from typing import Iterable, Optional


def _truncate_text(s: str, max_chars: int) -> str:
    """Soft cap a string to max_chars, ending on a word boundary if possible."""
    if len(s) <= max_chars:
        return s
    head = s[:max_chars]
    cut = head.rsplit(' ', 1)[0]
    return cut + f"\n... [{len(s) - len(cut)} chars truncated]"


def _truncate_tree(tree: str, max_leaves: int = 20) -> str:
    """Keep all subdir lines + first `max_leaves` leaf-file lines.

    Some tasks (BBBC002, RGI60 glaciers) list 50-200 image / archive files
    in their folder tree. The agent does not need every filename — it
    needs the shape (which subdirs exist, sample of files). We keep up to
    max_leaves leaves total, then emit "... N more files truncated".
    """
    lines = tree.splitlines()
    out: list[str] = []
    leaves_kept = 0
    leaves_total = sum(
        1 for ln in lines
        if ln.lstrip('|').lstrip('-').strip()
        and not ln.lstrip('|').lstrip('-').strip().endswith('/')
    )
    for line in lines:
        body = line.lstrip('|').lstrip('-').strip()
        if not body:
            out.append(line)
            continue
        is_subdir = body.endswith('/')
        if is_subdir:
            out.append(line)
            continue
        # leaf
        if leaves_kept < max_leaves:
            out.append(line)
            leaves_kept += 1
        elif leaves_kept == max_leaves:
            out.append(f"... [{leaves_total - max_leaves} more files truncated]")
            leaves_kept += 1  # sentinel so we don't repeat the message
    return "\n".join(out)


def _build_instruction(row: dict) -> str:
    """Compose the user-facing instruction from CSV columns.

    The order mirrors the upstream baselines so our results stay comparable.

    Truncations applied to keep prompts under ~16K tokens for Qwen3-8B
    (which has max_position_embeddings=40960 split with the response budget):
      - domain_knowledge: 4000 chars (~1K tokens)
      - dataset_folder_tree: 20 leaf files (image/archive directories can list
        50-200+ filenames otherwise; agent doesn't need every name)
      - dataset_preview: 3000 chars (~750 tokens; large CSV/h5 previews
        otherwise dominate the prompt)
    """
    parts = []
    parts.append(row.get('task_inst', '').strip())

    dk = (row.get('domain_knowledge') or '').strip()
    if dk:
        parts.append("# Domain Knowledge\n" + _truncate_text(dk, 4000))

    tree = (row.get('dataset_folder_tree') or '').strip()
    if tree:
        parts.append("# Dataset Folder Tree\n" + _truncate_tree(tree, max_leaves=20))

    preview = (row.get('dataset_preview') or '').strip()
    if preview:
        parts.append("# Dataset Preview\n" + _truncate_text(preview, 3000))

    return "\n\n".join(parts).strip()


def _build_input_files(row: dict, benchmark_dir: Optional[str]) -> list[tuple[str, str]]:
    """Resolve input files referenced by this task to (abs_path, rel_path) pairs.

    Strategy: parse the `dataset_folder_tree` string for `|-- ... |---- foo`
    leaf lines and join them against `benchmark_dir/datasets/`. If
    `benchmark_dir` is None (smoke / pre-download phase), return [].

    The `rel_path` (e.g. "dkpes/dkpes_train.csv") is the path the task
    instruction's folder tree and gold program reference, so the env MUST
    place the file at `workdir/rel_path` — NOT flatten it to the basename, or
    the agent's `open("dkpes/dkpes_train.csv")` fails.
    """
    if not benchmark_dir:
        return []

    datasets_dir = os.path.join(benchmark_dir, "datasets")
    if not os.path.isdir(datasets_dir):
        return []

    tree = row.get('dataset_folder_tree') or ''
    files: list[tuple[str, str]] = []
    # stack[depth] = subdir name at that depth (1-indexed).
    # '|--' = depth 1, '|----' = depth 2, '|------' = depth 3, etc.
    # Some tasks (e.g., BBBC002 image dir, RGI60 glacier archives) nest 4-5
    # levels deep, so we need to track the full path stack, not just one
    # current_subdir.
    stack: dict[int, str] = {}
    line_re = re.compile(r'^\|(-+)\s*(.*)$')
    for raw_line in tree.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith('|--'):
            continue
        m = line_re.match(line)
        if not m:
            continue
        depth = len(m.group(1)) // 2
        name = m.group(2).strip()
        is_subdir = name.endswith('/')
        name = name.rstrip('/')
        if not name:
            continue
        if is_subdir:
            stack[depth] = name
            # Drop any deeper subdir state that no longer applies.
            for d in [k for k in stack if k > depth]:
                del stack[d]
            continue
        # Leaf file: build the path from stack entries at depths < this leaf's,
        # then append the leaf name.
        parts = [stack[d] for d in sorted(stack) if d < depth]
        parts.append(name)
        rel = os.path.join(*parts) if parts else name
        abs_path = os.path.join(datasets_dir, rel)
        if os.path.exists(abs_path):
            files.append((abs_path, rel))
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
            pairs = _build_input_files(row, benchmark_dir)
            input_files = [abs_path for abs_path, _ in pairs]
            input_rel_paths = [rel for _, rel in pairs]
            expected = (row.get('output_fname') or '').strip() or None

            out.append({
                'task_id': str(iid),
                'instruction': instruction,
                'input_files': input_files,
                'input_rel_paths': input_rel_paths,
                'expected_output': expected,
                'workflow': workflow,
                # Absolute benchmark root so the env can locate this task's
                # eval_programs/<eval_script_name> at reward time (real-eval
                # scoring). None in smoke / pre-download phase.
                'benchmark_dir': (os.path.abspath(benchmark_dir)
                                  if benchmark_dir else None),
                'domain': (row.get('domain') or '').strip(),
                'eval_script_name': (row.get('eval_script_name') or '').strip(),
                'gold_program_name': (row.get('gold_program_name') or '').strip(),
                'subtask_categories': (row.get('subtask_categories') or '').strip(),
                'github_name': (row.get('github_name') or '').strip(),
            })
    return out
