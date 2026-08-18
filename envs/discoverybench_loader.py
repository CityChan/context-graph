"""Load official DiscoveryBench tasks into the code-agent row schema.

The upstream dataset is a directory tree rather than a single table.  Each
metadata file describes one or more natural-language queries and names the
data files located beside it.  Gold hypotheses live in separate answer-key
CSVs, which lets this loader keep them in ``extra_info`` for scoring without
placing them in the model prompt.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"cannot decode {path}")


def _read_json(path: Path) -> dict:
    return json.loads(_read_text(path))


def _flatten_queries(raw_queries: Iterable) -> Iterable[dict]:
    for group in raw_queries or []:
        if isinstance(group, dict):
            yield group
        else:
            yield from group


def _answer_key_path(root: Path, dataset_type: str) -> Path:
    candidates = (
        root / "answer_key" / f"answer_key_{dataset_type}.csv",
        root / "eval" / f"answer_key_{dataset_type}.csv",
        root / f"answer_key_{dataset_type}.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Missing answer_key_{dataset_type}.csv under {root}. "
        "Use the complete allenai/discoverybench snapshot."
    )


def _dataset_root(root: Path) -> Path:
    nested = root / "discoverybench"
    return nested if nested.is_dir() else root


def _load_answers(root: Path, dataset_type: str) -> dict[tuple[str, int, int], str]:
    text = _read_text(_answer_key_path(root, dataset_type))
    answers: dict[tuple[str, int, int], str] = {}
    for row in csv.DictReader(text.splitlines()):
        key = (row["dataset"], int(row["metadataid"]), int(row["query_id"]))
        answers[key] = row["gold_hypo"].strip()
    return answers


def _metadata_summary(metadata: dict, include_domain_knowledge: bool,
                      include_workflow_tags: bool) -> str:
    lines = []
    if metadata.get("domain"):
        lines.append(f"Domain: {metadata['domain']}")
    if include_workflow_tags and metadata.get("workflow_tags"):
        lines.append(f"Suggested workflow tags: {metadata['workflow_tags']}")
    if include_domain_knowledge and metadata.get("domain_knowledge"):
        lines.append(f"Domain knowledge: {metadata['domain_knowledge']}")
    lines.append("Dataset metadata:")
    for dataset in metadata.get("datasets", []):
        lines.append(f"- {dataset.get('name', 'data file')}: {dataset.get('description', '')}")
        columns = dataset.get("columns", [])
        if isinstance(columns, dict):
            columns = columns.get("raw", [])
        rendered = [
            f"{column.get('name', '')} ({column.get('description', '')})"
            for column in columns
        ]
        if rendered:
            lines.append("  Columns: " + "; ".join(rendered))
    return "\n".join(lines)


def load_discoverybench_tasks(
    benchmark_dir: str,
    dataset_type: str = "real",
    split: str = "test",
    workflow: str = "code",
    include_domain_knowledge: bool = False,
    include_workflow_tags: bool = False,
    task_ids: set[str] | None = None,
) -> list[dict]:
    """Return one task per query from an official dataset snapshot."""
    if dataset_type not in {"real", "synth"}:
        raise ValueError("dataset_type must be 'real' or 'synth'")

    root = Path(benchmark_dir).expanduser().resolve()
    dataset_root = _dataset_root(root)
    split_dir = dataset_root / dataset_type / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"DiscoveryBench split directory not found: {split_dir}")

    answers = _load_answers(root, dataset_type)
    tasks = []
    for metadata_path in sorted(split_dir.rglob("metadata_*.json")):
        metadata = _read_json(metadata_path)
        dataset_name = metadata_path.parent.name
        # The official real split contains metadata files whose internal `id`
        # field is stale (for example metadata_19.json declares id=0). The
        # answer key indexes the filename id, so that is authoritative.
        metadata_id = int(metadata_path.stem.rsplit("_", 1)[-1])

        input_files = []
        input_rel_paths = []
        for dataset in metadata.get("datasets", []):
            rel = str(dataset.get("name", "")).strip()
            if not rel:
                continue
            source = metadata_path.parent / rel
            if not source.is_file():
                raise FileNotFoundError(
                    f"Metadata {metadata_path} references missing dataset file {source}"
                )
            input_files.append(str(source.resolve()))
            input_rel_paths.append(rel)

        metadata_text = _metadata_summary(
            metadata, include_domain_knowledge, include_workflow_tags
        )
        for query in _flatten_queries(metadata.get("queries", [])):
            qid = int(query["qid"])
            task_id = f"{dataset_type}:{dataset_name}:m{metadata_id}:q{qid}"
            if task_ids is not None and task_id not in task_ids:
                continue
            answer_key = (dataset_name, metadata_id, qid)
            if answer_key not in answers:
                raise KeyError(f"No gold hypothesis for {answer_key}")

            question = str(query["question"]).strip()
            instruction = (
                "Analyze the provided dataset(s) and answer the discovery question below.\n\n"
                f"Discovery question: {question}\n\n"
                f"{metadata_text}\n\n"
                "Use Python to inspect and analyze the data. Your final hypothesis must "
                "directly answer the question and preserve quantitative values, boundary "
                "conditions, directions, and relationship forms supported by the data. "
                "Also describe the analysis workflow actually used. Save exactly one JSON "
                "object to pred_results/discovery_result.json with non-empty string fields "
                "`hypothesis` and `workflow`."
            )
            tasks.append({
                "task_id": task_id,
                "instruction": instruction,
                "query": question,
                "input_files": input_files,
                "input_rel_paths": input_rel_paths,
                "expected_output": "discovery_result.json",
                "workflow": workflow,
                "benchmark_dir": str(root),
                "metadata_path": str(metadata_path.resolve()),
                "metadata": metadata,
                "dataset_type": dataset_type,
                "dataset_split": split,
                "dataset_name": dataset_name,
                "metadata_id": metadata_id,
                "query_id": qid,
                "query_type": query.get("question_type", "unknown"),
                "difficulty": query.get("difficulty"),
                "gold_hypothesis": answers[answer_key],
                "gold_workflow": "",
                "domain": metadata.get("domain", "unknown"),
            })
    return tasks
