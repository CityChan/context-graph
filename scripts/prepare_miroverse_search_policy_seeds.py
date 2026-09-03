#!/usr/bin/env python3
"""Extract deduplicated MiroVerse questions and exact answers for native rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.inspect_miroverse_policy_source import as_messages, load_records
from scripts.prepare_miroverse_contextgraph_policy_sft import (
    boxed_answer,
    collapse_finalizer_round,
)


LATEX_TEXT_PATTERN = re.compile(r"\\(?:text|textrm|mathrm|operatorname)\{([^{}]*)\}")


def normalize_miroverse_answer(answer: str) -> str:
    """Convert common textual LaTeX wrappers and spacing into judgeable text."""
    normalized = answer.strip().strip("$")
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = LATEX_TEXT_PATTERN.sub(r"\1", normalized)
    normalized = re.sub(r"\\(?:,|;|:|!|quad|qquad)", " ", normalized)
    normalized = normalized.replace(r"\ ", " ").replace("~", " ")
    normalized = re.sub(r"\\([#$%&_{}])", r"\1", normalized)
    return " ".join(normalized.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    return parser.parse_args()


def record_to_seed(
    record,
    index: int,
    reasons: list[str] | None = None,
    *,
    source_subset: str = "MiroVerse-MuSiQue",
):
    def reject(reason: str):
        if reasons is not None:
            reasons.append(reason)

    original_messages = as_messages(record.get("messages", record.get("conversations")))
    messages, finalizer_collapsed = collapse_finalizer_round(original_messages)
    if len(messages) < 3 or messages[1]["role"] != "user" or messages[-1]["role"] != "assistant":
        reject("message_shape")
        return None
    query = messages[1]["content"].strip()
    # MiroVerse often appends an artificial finalizer round whose assistant
    # response contains the normalized boxed gold answer. Remove that round
    # from the rollout context, but retain its answer for reward supervision.
    answer_sources = []
    if finalizer_collapsed:
        answer_sources.append(original_messages[-1]["content"])
    answer_sources.append(messages[-1]["content"])
    answer = next((value for text in answer_sources if (value := boxed_answer(text))), None)
    if not query:
        reject("empty_query")
        return None
    if not answer:
        reject("missing_boxed_answer")
        return None
    answer = normalize_miroverse_answer(answer)
    if not answer:
        reject("empty_normalized_answer")
        return None
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    subset_slug = re.sub(r"[^a-z0-9]+", "_", source_subset.lower()).strip("_")
    subset_slug = re.sub(r"^miroverse_", "", subset_slug)
    task_id = f"miroverse_{subset_slug}_{query_hash[:16]}"
    extra = {
        "task_id": task_id,
        "instance_id": task_id,
        "query": query,
        "problem_statement": query,
        "answer": answer,
        "workflow": "search_graph",
        "level": "miroverse",
        "source": f"miromind-ai/MiroVerse-v0.1:{source_subset}",
        "source_subset": source_subset,
        "source_index": index,
        "query_hash": query_hash,
    }
    return {
        "prompt": [{"role": "user", "content": query}],
        # ability controls environment dispatch; dataset identity belongs in
        # data_source and extra_info.source.
        "ability": "LocalSearch",
        "data_source": f"miroverse_{subset_slug}",
        "extra_info": extra,
        "reward_model": {"style": "rule", "ground_truth": answer},
    }


def build_seeds(records, max_samples: int):
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    seeds = {}
    counters = Counter()
    for index, record in enumerate(records):
        if max_samples and counters["input"] >= max_samples:
            break
        counters["input"] += 1
        reasons = []
        source_subset = str(record.get("_miroverse_source_subset", "MiroVerse-MuSiQue"))
        source_index = int(record.get("_miroverse_source_index", index))
        seed = record_to_seed(
            record,
            source_index,
            reasons,
            source_subset=source_subset,
        )
        if seed is None:
            counters[f"rejected:{reasons[0] if reasons else 'unknown'}"] += 1
            continue
        query_hash = seed["extra_info"]["query_hash"]
        if query_hash in seeds:
            counters["duplicates"] += 1
            continue
        seeds[query_hash] = seed
        counters["accepted"] += 1
    return list(seeds.values()), counters


def source_records(paths: Iterable[Path]):
    for path in paths:
        subset = path.stem
        for source_index, record in enumerate(load_records(path)):
            annotated = dict(record)
            annotated["_miroverse_source_subset"] = subset
            annotated["_miroverse_source_index"] = source_index
            yield annotated


def main() -> None:
    args = parse_args()
    input_paths = [Path(value) for value in args.input]
    seeds, counters = build_seeds(source_records(input_paths), args.max_samples)
    if not seeds:
        raise SystemExit(f"no MiroVerse policy seeds accepted: {dict(counters)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seeds).to_parquet(output, index=False)
    manifest = {
        "schema_version": "contextgraph.miroverse_policy_seeds.v1",
        "input": (
            str(input_paths[0])
            if len(input_paths) == 1
            else [str(path) for path in input_paths]
        ),
        "output": str(output),
        "ability": "LocalSearch",
        "data_source": (
            seeds[0]["data_source"]
            if len(input_paths) == 1
            else "miroverse_multi_subset"
        ),
        **dict(counters),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
