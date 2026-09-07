#!/usr/bin/env python3
"""Prepare the canonical GSM8K Parquet format consumed by VERL GRPO."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FINAL_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$")


def extract_ground_truth(answer: str) -> str:
    match = FINAL_ANSWER_RE.search(answer)
    if match is None:
        raise ValueError(f"GSM8K answer has no final '####' value: {answer[-120:]!r}")
    return match.group(1).replace(",", "")


def convert_example(example: dict, split: str, index: int) -> dict:
    question = str(example["question"]).strip()
    answer = str(example["answer"])
    return {
        "data_source": "openai/gsm8k",
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"{question}\n\nSolve the problem step by step and put the final "
                    'numeric answer after "####".'
                ),
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": extract_ground_truth(answer),
        },
        "extra_info": {"split": split, "index": index},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import Dataset, load_dataset

    dataset = load_dataset("openai/gsm8k", "main")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source_split, output_name in (("train", "train.parquet"), ("test", "test.parquet")):
        rows = [
            convert_example(example, source_split, index)
            for index, example in enumerate(dataset[source_split])
        ]
        output_path = args.output_dir / output_name
        Dataset.from_list(rows).to_parquet(str(output_path))
        print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
