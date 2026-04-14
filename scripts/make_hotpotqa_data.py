#!/usr/bin/env python3
"""Download HotpotQA dataset from HuggingFace and convert to FoldAgent parquet format.

HotpotQA: 2-hop multi-hop QA requiring reasoning across two Wikipedia paragraphs.
This is a meaningful benchmark for ContextGraph since the agent must:
  - Search for information about entity A
  - Search for information about entity B
  - Combine findings to answer a bridging/comparison question

Example: "Were Scott Derrickson and Ed Wood of the same nationality?"
  → Need to find Scott Derrickson's nationality AND Ed Wood's nationality → compare.

HotpotQA distractor setting includes 10 paragraphs (2 gold + 8 distractors),
but we use the fullwiki setting (no context given) so the agent must search.

Each row format:
  prompt: [{"role": "user", "content": question}]
  ability: "LocalSearch@hotpotqa"
  extra_info:
    query: question text
    answer: canonical answer
    problem_statement: question text
    type: bridge / comparison
    level: easy / medium / hard
    workflow: configurable
"""

import os
import argparse
import pandas as pd


def load_hotpotqa(n_train=500, n_val=100, seed=42):
    """Load HotpotQA from HuggingFace datasets library."""
    from datasets import load_dataset
    print("Loading HotpotQA from HuggingFace (hotpotqa/hotpot_qa, distractor config)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

    train = ds["train"].shuffle(seed=seed).select(range(min(n_train, len(ds["train"]))))
    val = ds["validation"].shuffle(seed=seed).select(range(min(n_val, len(ds["validation"]))))

    print(f"Loaded {len(train)} train + {len(val)} val samples")
    return train, val


def to_row(item, workflow):
    """Convert a HotpotQA item to a FoldAgent parquet row."""
    question = item["question"]
    answer = item["answer"]

    return {
        "prompt": [{"role": "user", "content": question}],
        "ability": "LocalSearch@hotpotqa",
        "extra_info": {
            "query": question,
            "answer": answer,
            "problem_statement": question,
            "type": item.get("type", ""),
            "level": item.get("level", ""),
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train, val = load_hotpotqa(args.n_train, args.n_val, args.seed)

    # Print type/level distribution
    from collections import Counter
    train_types = Counter(item.get("type", "?") for item in train)
    train_levels = Counter(item.get("level", "?") for item in train)
    print(f"Train type distribution: {dict(train_types)}")
    print(f"Train level distribution: {dict(train_levels)}")

    # FoldAgent (search_branch) version
    train_rows = [to_row(item, "search_branch") for item in train]
    val_rows = [to_row(item, "search_branch") for item in val]
    pd.DataFrame(train_rows).to_parquet(f"{args.out_dir}/hotpotqa_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{args.out_dir}/hotpotqa_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/hotpotqa_train.parquet ({len(train_rows)} rows, workflow=search_branch)")
    print(f"Wrote {args.out_dir}/hotpotqa_test.parquet ({len(val_rows)} rows, workflow=search_branch)")

    # ContextGraph (search_graph) version
    train_rows_g = [to_row(item, "search_graph") for item in train]
    val_rows_g = [to_row(item, "search_graph") for item in val]
    pd.DataFrame(train_rows_g).to_parquet(f"{args.out_dir}/hotpotqa_graph_train.parquet", index=False)
    pd.DataFrame(val_rows_g).to_parquet(f"{args.out_dir}/hotpotqa_graph_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/hotpotqa_graph_train.parquet ({len(train_rows_g)} rows, workflow=search_graph)")
    print(f"Wrote {args.out_dir}/hotpotqa_graph_test.parquet ({len(val_rows_g)} rows, workflow=search_graph)")

    # Print sample rows
    print("\nSample rows:")
    for i in range(min(5, len(train_rows))):
        row = train_rows[i]
        print(f"  [{row['extra_info']['type']}/{row['extra_info']['level']}] Q: {row['extra_info']['query']}")
        print(f"    A: {row['extra_info']['answer']}")


if __name__ == "__main__":
    main()
