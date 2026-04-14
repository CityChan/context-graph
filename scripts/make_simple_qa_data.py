#!/usr/bin/env python3
"""Generate simple QA dataset for FoldAgent / ContextGraph training tests.

Uses well-known facts that even a 4B model should know without search.
This produces non-zero rewards in early RL steps so training has signal.

Train: 32 simple factual questions (split into 4 batches of 8)
Val: 16 questions (subset of train, for quick validation)
"""

import os
import pandas as pd

SIMPLE_QA = [
    # Capitals
    {"query": "What is the capital of France?", "answer": "Paris"},
    {"query": "What is the capital of Japan?", "answer": "Tokyo"},
    {"query": "What is the capital of Italy?", "answer": "Rome"},
    {"query": "What is the capital of Germany?", "answer": "Berlin"},
    {"query": "What is the capital of Spain?", "answer": "Madrid"},
    {"query": "What is the capital of Russia?", "answer": "Moscow"},
    {"query": "What is the capital of China?", "answer": "Beijing"},
    {"query": "What is the capital of Australia?", "answer": "Canberra"},
    # Famous people
    {"query": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"query": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"query": "Who developed the theory of relativity?", "answer": "Albert Einstein"},
    {"query": "Who was the first president of the United States?", "answer": "George Washington"},
    {"query": "Who wrote the novel 1984?", "answer": "George Orwell"},
    {"query": "Who invented the telephone?", "answer": "Alexander Graham Bell"},
    {"query": "Who composed the Ninth Symphony?", "answer": "Ludwig van Beethoven"},
    {"query": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    # Science / nature
    {"query": "What is the largest planet in the solar system?", "answer": "Jupiter"},
    {"query": "What is the smallest planet in the solar system?", "answer": "Mercury"},
    {"query": "What is the chemical symbol for gold?", "answer": "Au"},
    {"query": "What is the chemical symbol for water?", "answer": "H2O"},
    {"query": "How many continents are there on Earth?", "answer": "7"},
    {"query": "What is the tallest mountain in the world?", "answer": "Mount Everest"},
    {"query": "What is the longest river in the world?", "answer": "Nile"},
    {"query": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    # History
    {"query": "What year did World War II end?", "answer": "1945"},
    {"query": "What year did the Berlin Wall fall?", "answer": "1989"},
    {"query": "Who was the first person to walk on the Moon?", "answer": "Neil Armstrong"},
    {"query": "In what year did the Titanic sink?", "answer": "1912"},
    # Sports / pop culture
    {"query": "How many players are on a soccer team on the field?", "answer": "11"},
    {"query": "What sport is known as the king of sports?", "answer": "Soccer"},
    {"query": "Who is the main character in the Harry Potter series?", "answer": "Harry Potter"},
    {"query": "What is the name of the wizard school in Harry Potter?", "answer": "Hogwarts"},
]


def make_row(item, workflow):
    prompt = [{"role": "user", "content": item["query"]}]
    return {
        "prompt": prompt,
        "ability": "LocalSearch@simple_qa",
        "extra_info": {
            "query": item["query"],
            "answer": item["answer"],
            "problem_statement": item["query"],
            "workflow": workflow,
        },
    }


def main():
    os.makedirs("data", exist_ok=True)

    # FoldAgent (search_branch) version
    train_rows = [make_row(item, "search_branch") for item in SIMPLE_QA]
    val_rows = [make_row(item, "search_branch") for item in SIMPLE_QA[:16]]

    pd.DataFrame(train_rows).to_parquet("data/simple_qa_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet("data/simple_qa_test.parquet", index=False)
    print(f"Created data/simple_qa_train.parquet ({len(train_rows)} rows, workflow=search_branch)")
    print(f"Created data/simple_qa_test.parquet ({len(val_rows)} rows, workflow=search_branch)")

    # ContextGraph (search_graph) version
    train_rows_g = [make_row(item, "search_graph") for item in SIMPLE_QA]
    val_rows_g = [make_row(item, "search_graph") for item in SIMPLE_QA[:16]]

    pd.DataFrame(train_rows_g).to_parquet("data/simple_qa_graph_train.parquet", index=False)
    pd.DataFrame(val_rows_g).to_parquet("data/simple_qa_graph_test.parquet", index=False)
    print(f"Created data/simple_qa_graph_train.parquet ({len(train_rows_g)} rows, workflow=search_graph)")
    print(f"Created data/simple_qa_graph_test.parquet ({len(val_rows_g)} rows, workflow=search_graph)")


if __name__ == "__main__":
    main()
