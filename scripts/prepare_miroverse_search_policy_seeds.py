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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all rows")
    return parser.parse_args()


def record_to_seed(record, index: int, reasons: list[str] | None = None):
    def reject(reason: str):
        if reasons is not None:
            reasons.append(reason)

    messages = as_messages(record.get("messages", record.get("conversations")))
    messages, _ = collapse_finalizer_round(messages)
    if len(messages) < 3 or messages[1]["role"] != "user" or messages[-1]["role"] != "assistant":
        reject("message_shape")
        return None
    query = messages[1]["content"].strip()
    answer = boxed_answer(messages[-1]["content"])
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
    task_id = f"miroverse_musique_{query_hash[:16]}"
    extra = {
        "task_id": task_id,
        "instance_id": task_id,
        "query": query,
        "problem_statement": query,
        "answer": answer,
        "workflow": "search_graph",
        "level": "miroverse",
        "source": "miromind-ai/MiroVerse-v0.1:MiroVerse-MuSiQue",
        "source_index": index,
        "query_hash": query_hash,
    }
    return {
        "prompt": [{"role": "user", "content": query}],
        "ability": "miroverse_musique",
        "data_source": "miroverse_musique",
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
        seed = record_to_seed(record, index, reasons)
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


def main() -> None:
    args = parse_args()
    seeds, counters = build_seeds(load_records(Path(args.input)), args.max_samples)
    if not seeds:
        raise SystemExit(f"no MiroVerse policy seeds accepted: {dict(counters)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(seeds).to_parquet(output, index=False)
    manifest = {
        "schema_version": "contextgraph.miroverse_policy_seeds.v1",
        "input": args.input,
        "output": str(output),
        **dict(counters),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
