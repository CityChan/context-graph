#!/usr/bin/env python3
"""Generate minimal dummy parquet files for FoldAgent smoke test."""

import os
import pandas as pd

DUMMY_ITEMS = [
    {
        "query": "What is the capital of France?",
        "answer": "Paris",
        "data_source": "dummy",
        "difficulty": "easy",
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare",
        "data_source": "dummy",
        "difficulty": "easy",
    },
    {
        "query": "What is the largest planet in the solar system?",
        "answer": "Jupiter",
        "data_source": "dummy",
        "difficulty": "easy",
    },
    {
        "query": "What year did World War II end?",
        "answer": "1945",
        "data_source": "dummy",
        "difficulty": "easy",
    },
]


def make_row(item):
    prompt = [{"role": "user", "content": item["query"]}]
    return {
        "prompt": prompt,
        "ability": f"LocalSearch@dummy",
        "extra_info": {
            "query": item["query"],
            "answer": item["answer"],
            "problem_statement": item["query"],
            "workflow": "search",
        },
    }


def main():
    os.makedirs("data", exist_ok=True)

    rows = [make_row(item) for item in DUMMY_ITEMS]
    df = pd.DataFrame(rows)

    df.to_parquet("data/dummy_train.parquet", index=False)
    df.to_parquet("data/dummy_test.parquet", index=False)
    print(f"Created data/dummy_train.parquet ({len(df)} rows)")
    print(f"Created data/dummy_test.parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()
