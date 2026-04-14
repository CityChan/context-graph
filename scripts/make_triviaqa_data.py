#!/usr/bin/env python3
"""Download TriviaQA dataset from HuggingFace and convert to FoldAgent parquet format.

TriviaQA: simple factual questions with short answers (good EM evaluation).
- Train: ~78k samples (we subsample for speed)
- Val: ~11k samples (we subsample for speed)

Each row format:
  prompt: [{"role": "user", "content": question}]
  ability: "LocalSearch@triviaqa"
  extra_info:
    query: question text
    answer: canonical answer (normalized_value)
    problem_statement: question text
    workflow: configurable
"""

import os
import argparse
import pandas as pd


def load_triviaqa(n_train=200, n_val=64, seed=42):
    """Load TriviaQA from HuggingFace datasets library."""
    from datasets import load_dataset
    print("Loading TriviaQA from HuggingFace (mandarjoshi/trivia_qa, rc.nocontext config)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

    train = ds["train"].shuffle(seed=seed).select(range(min(n_train, len(ds["train"]))))
    val = ds["validation"].shuffle(seed=seed).select(range(min(n_val, len(ds["validation"]))))

    print(f"Loaded {len(train)} train + {len(val)} val samples")
    return train, val


def to_row(item, workflow):
    """Convert a TriviaQA item to a FoldAgent parquet row."""
    question = item["question"]
    # TriviaQA answer has many fields; use normalized_value (lowercase, canonical)
    answer = item["answer"].get("normalized_value") or item["answer"]["value"]

    return {
        "prompt": [{"role": "user", "content": question}],
        "ability": "LocalSearch@triviaqa",
        "extra_info": {
            "query": question,
            "answer": answer,
            "problem_statement": question,
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_val", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train, val = load_triviaqa(args.n_train, args.n_val, args.seed)

    # FoldAgent (search_branch) version
    train_rows = [to_row(item, "search_branch") for item in train]
    val_rows = [to_row(item, "search_branch") for item in val]
    pd.DataFrame(train_rows).to_parquet(f"{args.out_dir}/triviaqa_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{args.out_dir}/triviaqa_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/triviaqa_train.parquet ({len(train_rows)} rows, workflow=search_branch)")
    print(f"Wrote {args.out_dir}/triviaqa_test.parquet ({len(val_rows)} rows, workflow=search_branch)")

    # ContextGraph (search_graph) version
    train_rows_g = [to_row(item, "search_graph") for item in train]
    val_rows_g = [to_row(item, "search_graph") for item in val]
    pd.DataFrame(train_rows_g).to_parquet(f"{args.out_dir}/triviaqa_graph_train.parquet", index=False)
    pd.DataFrame(val_rows_g).to_parquet(f"{args.out_dir}/triviaqa_graph_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/triviaqa_graph_train.parquet ({len(train_rows_g)} rows, workflow=search_graph)")
    print(f"Wrote {args.out_dir}/triviaqa_graph_test.parquet ({len(val_rows_g)} rows, workflow=search_graph)")

    # Print 3 sample rows for sanity check
    print("\nSample rows:")
    for i in range(min(3, len(train_rows))):
        print(f"  Q: {train_rows[i]['extra_info']['query']}")
        print(f"  A: {train_rows[i]['extra_info']['answer']}")


if __name__ == "__main__":
    main()
