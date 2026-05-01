#!/usr/bin/env python3
"""Generate HotpotQA parquet for FoldAgent / ContextGraph training (fullwiki setting).

Downloads HotpotQA from HuggingFace (`hotpot_qa` repo, `distractor` config:
the questions are identical to fullwiki, but distractor carries the
supporting_facts annotation we keep for evaluation).

Outputs four parquet files matching the schema make_multihop_data.py /
make_alfworld_data.py emit, so the same trainer entry points can be used:
  data/hotpotqa_train.parquet        (workflow=search_branch, FoldAgent)
  data/hotpotqa_test.parquet
  data/hotpotqa_graph_train.parquet  (workflow=search_graph,  ContextGraph)
  data/hotpotqa_graph_test.parquet

The Wikipedia corpus the agent searches against is NOT bundled here — it is
served by envs/search_server.py against an external HotpotQA Wiki dump
(see scripts/build_hotpotqa_index.* once that lands).

Note: HuggingFace dataset download needs internet. Run this on a Vista login
node, not on a compute node; once parquet + HF cache are populated, the
training nodes can read offline.

Usage:
  python scripts/make_hotpotqa_data.py --n_train 2000 --n_val 500
  python scripts/make_hotpotqa_data.py --all              # full 90K train, 7K val
"""

import os
import argparse
import random
from collections import Counter

import pandas as pd


def to_row(item, workflow):
    # supporting_facts in HF schema is a parallel-array dict.
    # Convert to list of [title, sent_id] for downstream readability.
    sf = item.get("supporting_facts", {})
    if isinstance(sf, dict):
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])
        supporting = [[t, s] for t, s in zip(titles, sent_ids)]
    else:
        supporting = sf

    return {
        "prompt": [{"role": "user", "content": item["question"]}],
        "ability": "LocalSearch@hotpotqa",
        "extra_info": {
            "query": item["question"],
            "answer": item["answer"],
            "problem_statement": item["question"],
            "type": item.get("type", "unknown"),
            "level": item.get("level", "unknown"),
            "supporting_facts": supporting,
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--all", action="store_true",
                        help="Use the full HotpotQA train (~90K) + validation (~7K) splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--cache_dir", default=None,
                        help="HF datasets cache dir (defaults to $HF_HOME/datasets or ~/.cache/huggingface).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "`datasets` not installed. `pip install datasets` (should be in cxtgraph env)."
        ) from e

    print("Loading HotpotQA (distractor config) from HuggingFace...")
    ds = load_dataset(
        "hotpot_qa",
        "distractor",
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    print(f"Loaded splits: {list(ds.keys())}")
    print(f"  train: {len(ds['train'])}  validation: {len(ds['validation'])}")

    train_items = list(ds["train"])
    val_items = list(ds["validation"])

    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    if not args.all:
        train_items = train_items[: args.n_train]
        val_items = val_items[: args.n_val]

    print(f"Using train={len(train_items)}  val={len(val_items)}")

    type_counts = Counter(t.get("type", "?") for t in train_items)
    level_counts = Counter(t.get("level", "?") for t in train_items)
    print(f"Train types:  {dict(type_counts)}")
    print(f"Train levels: {dict(level_counts)}")

    # FoldAgent (search_branch)
    train_rows = [to_row(t, "search_branch") for t in train_items]
    val_rows = [to_row(t, "search_branch") for t in val_items]
    pd.DataFrame(train_rows).to_parquet(f"{args.out_dir}/hotpotqa_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{args.out_dir}/hotpotqa_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/hotpotqa_train.parquet  ({len(train_rows)} rows, search_branch)")
    print(f"Wrote {args.out_dir}/hotpotqa_test.parquet   ({len(val_rows)} rows, search_branch)")

    # ContextGraph (search_graph)
    train_rows_g = [to_row(t, "search_graph") for t in train_items]
    val_rows_g = [to_row(t, "search_graph") for t in val_items]
    pd.DataFrame(train_rows_g).to_parquet(f"{args.out_dir}/hotpotqa_graph_train.parquet", index=False)
    pd.DataFrame(val_rows_g).to_parquet(f"{args.out_dir}/hotpotqa_graph_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/hotpotqa_graph_train.parquet  ({len(train_rows_g)} rows, search_graph)")
    print(f"Wrote {args.out_dir}/hotpotqa_graph_test.parquet   ({len(val_rows_g)} rows, search_graph)")

    print("\nSample questions:")
    for t in train_items[:5]:
        print(f"  [{t.get('type', '?')}/{t.get('level', '?')}] Q: {t['question']}")
        print(f"    A: {t['answer']}")


if __name__ == "__main__":
    main()
