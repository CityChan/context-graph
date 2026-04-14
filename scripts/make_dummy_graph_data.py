#!/usr/bin/env python3
"""Generate dummy parquet files for ContextGraph smoke test.

Same as make_dummy_data.py but with workflow=search_graph.
"""

import os
import pandas as pd

DUMMY_ITEMS = [
    {
        "query": "What is the capital of France?",
        "answer": "Paris",
    },
    {
        "query": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare",
    },
    {
        "query": "What is the largest planet in the solar system?",
        "answer": "Jupiter",
    },
    {
        "query": "What year did World War II end?",
        "answer": "1945",
    },
]


def make_row(item):
    prompt = [{"role": "user", "content": item["query"]}]
    return {
        "prompt": prompt,
        "ability": "LocalSearch@dummy",
        "extra_info": {
            "query": item["query"],
            "answer": item["answer"],
            "problem_statement": item["query"],
            "workflow": "search_graph",
        },
    }


def main():
    os.makedirs("data", exist_ok=True)

    rows = [make_row(item) for item in DUMMY_ITEMS]
    df = pd.DataFrame(rows)

    df.to_parquet("data/dummy_graph_train.parquet", index=False)
    df.to_parquet("data/dummy_graph_test.parquet", index=False)
    print(f"Created data/dummy_graph_train.parquet ({len(df)} rows)")
    print(f"Created data/dummy_graph_test.parquet ({len(df)} rows)")


if __name__ == "__main__":
    main()
