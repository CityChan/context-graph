#!/usr/bin/env python3
"""Generate 2WikiMultiHopQA parquet for FoldAgent / ContextGraph training.

Downloads 2WikiMultiHopQA from HuggingFace (default `xanhho/2WikiMultihopQA`
— override with --hf_repo). Schema mirrors make_hotpotqa_data.py so the
trainer entry points are reusable; outputs four parquet files:

  data/2wikimqa_train.parquet        (workflow=search_branch, FoldAgent)
  data/2wikimqa_test.parquet
  data/2wikimqa_graph_train.parquet  (workflow=search_graph,  ContextGraph)
  data/2wikimqa_graph_test.parquet

Compared to HotpotQA, every example carries an additional `evidences`
field — a list of (subject, relation, object) triples that form the
ground-truth reasoning chain. We persist it through extra_info so a
graph-aware reward can pick it up later. ContextGraph's flat/scope
process_reward will work with or without it; the graph reward channel
can use it for chain-accuracy supervision.

Usage:
  python scripts/make_2wikimqa_data.py --n_train 2000 --n_val 500
  python scripts/make_2wikimqa_data.py --all
  python scripts/make_2wikimqa_data.py --hf_repo voidful/2WikiMultihopQA
"""

import os
import argparse
import random
from collections import Counter

import pandas as pd


def _normalize_supporting_facts(sf):
    """Return a list of [title, str(sent_id)] pairs.

    HF parallel-array form: {"title": [...], "sent_id": [...]}
    Older list-of-dicts form: [{"title": ..., "sent_id": ...}, ...]
    Some 2WikiMQA mirrors use plain list of [title, sent_id].
    """
    if sf is None:
        return []
    if isinstance(sf, dict):
        titles = sf.get("title", [])
        sent_ids = sf.get("sent_id", [])
        return [[t, str(s)] for t, s in zip(titles, sent_ids)]
    out = []
    for entry in sf:
        if isinstance(entry, dict):
            out.append([entry.get("title", ""), str(entry.get("sent_id", 0))])
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            out.append([entry[0], str(entry[1])])
    return out


def _normalize_evidences(ev):
    """Return a list of [subj, rel, obj] string triples.

    HF parallel-array form: {"fact": [[s, r, o], ...]}  (xanhho's port)
    or {"subject": [...], "relation": [...], "object": [...]}
    or plain list of [s, r, o].
    """
    if ev is None:
        return []
    if isinstance(ev, dict):
        if "fact" in ev:
            facts = ev["fact"]
            return [[str(x[0]), str(x[1]), str(x[2])] for x in facts if len(x) >= 3]
        subjs = ev.get("subject", [])
        rels = ev.get("relation", [])
        objs = ev.get("object", [])
        return [[str(s), str(r), str(o)] for s, r, o in zip(subjs, rels, objs)]
    out = []
    for entry in ev:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            out.append([str(entry[0]), str(entry[1]), str(entry[2])])
    return out


def to_row(item, workflow):
    return {
        "prompt": [{"role": "user", "content": item["question"]}],
        "ability": "LocalSearch@2wikimqa",
        "extra_info": {
            "query": item["question"],
            "answer": item["answer"],
            "problem_statement": item["question"],
            "type": item.get("type", "unknown"),
            "level": item.get("level", "unknown"),
            "supporting_facts": _normalize_supporting_facts(item.get("supporting_facts")),
            "evidences": _normalize_evidences(item.get("evidences")),
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--all", action="store_true",
                        help="Use the full 2WikiMQA train (~167K) + dev (~12K) splits.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--hf_repo", default="xanhho/2WikiMultihopQA",
                        help="HF dataset repo (default: xanhho/2WikiMultihopQA).")
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

    print(f"Loading 2WikiMultiHopQA from HuggingFace: {args.hf_repo} ...")
    ds = load_dataset(args.hf_repo, trust_remote_code=True, cache_dir=args.cache_dir)
    print(f"Loaded splits: {list(ds.keys())}")

    # Common split names: train / dev / test (or validation)
    train_key = "train"
    val_key = "validation" if "validation" in ds else ("dev" if "dev" in ds else None)
    if val_key is None:
        raise SystemExit(f"No validation/dev split found in {args.hf_repo}: got {list(ds.keys())}")
    print(f"  {train_key}: {len(ds[train_key])}  {val_key}: {len(ds[val_key])}")

    train_items = list(ds[train_key])
    val_items = list(ds[val_key])

    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    if not args.all:
        train_items = train_items[: args.n_train]
        val_items = val_items[: args.n_val]

    print(f"Using train={len(train_items)}  val={len(val_items)}")

    type_counts = Counter(t.get("type", "?") for t in train_items)
    print(f"Train types:  {dict(type_counts)}")

    # FoldAgent (search_branch)
    train_rows = [to_row(t, "search_branch") for t in train_items]
    val_rows = [to_row(t, "search_branch") for t in val_items]
    pd.DataFrame(train_rows).to_parquet(f"{args.out_dir}/2wikimqa_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{args.out_dir}/2wikimqa_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/2wikimqa_train.parquet  ({len(train_rows)} rows, search_branch)")
    print(f"Wrote {args.out_dir}/2wikimqa_test.parquet   ({len(val_rows)} rows, search_branch)")

    # ContextGraph (search_graph)
    train_rows_g = [to_row(t, "search_graph") for t in train_items]
    val_rows_g = [to_row(t, "search_graph") for t in val_items]
    pd.DataFrame(train_rows_g).to_parquet(f"{args.out_dir}/2wikimqa_graph_train.parquet", index=False)
    pd.DataFrame(val_rows_g).to_parquet(f"{args.out_dir}/2wikimqa_graph_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/2wikimqa_graph_train.parquet  ({len(train_rows_g)} rows, search_graph)")
    print(f"Wrote {args.out_dir}/2wikimqa_graph_test.parquet   ({len(val_rows_g)} rows, search_graph)")

    print("\nSample questions:")
    for t in train_items[:5]:
        print(f"  [{t.get('type', '?')}] Q: {t['question']}")
        print(f"    A: {t['answer']}")
        ev = _normalize_evidences(t.get("evidences"))
        if ev:
            print(f"    chain: {' -> '.join('(' + ', '.join(e) + ')' for e in ev[:3])}")


if __name__ == "__main__":
    main()
