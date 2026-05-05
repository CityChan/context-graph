#!/usr/bin/env python3
"""Generate Bamboogle parquet for FoldAgent / ContextGraph evaluation.

Bamboogle (Press et al., 2022) is a small (~125 question) multi-hop QA
benchmark. Eval-only — no training split. Schema mirrors hotpotqa /
2wikimqa / musique data prep so the trainer entry points are reusable.

Outputs two parquet files (no train, eval-only):
  data/bamboogle_test.parquet        (workflow=search_branch, FoldAgent)
  data/bamboogle_graph_test.parquet  (workflow=search_graph,  ContextGraph)

Bamboogle has no distractor paragraphs — questions assume open-domain
retrieval over Wikipedia. The unified Wiki corpus (HotpotQA + 2WikiMQA
+ MuSiQue) should cover most entities; missing ones default to whatever
the embedding retriever finds nearest.

Usage:
  python scripts/make_bamboogle_data.py
  python scripts/make_bamboogle_data.py --hf_repo chiayewken/bamboogle
"""

import argparse
import json
import os

import pandas as pd


def _hf_parquet_load(repo, config=None, cache_dir=None):
    """Load HF dataset via auto-converted parquet refs (bypass loading scripts)."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset", revision="refs/convert/parquet")
    splits = {}
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        if len(parts) < 3:
            continue
        if config is not None and parts[0] != config:
            continue
        splits.setdefault(parts[-2], []).append(f)
    if not splits:
        raise RuntimeError(f"No parquet files at refs/convert/parquet for {repo}")
    result = {}
    for split, paths in splits.items():
        dfs = []
        for p in sorted(paths):
            local = hf_hub_download(repo, p, repo_type="dataset",
                                    revision="refs/convert/parquet", cache_dir=cache_dir)
            dfs.append(pd.read_parquet(local))
        result[split] = pd.concat(dfs, ignore_index=True).to_dict("records")
    return result


def _get_field(item, *names):
    """Robust field accessor — Bamboogle mirrors use varying capitalization."""
    for n in names:
        if n in item and item[n] is not None:
            return item[n]
    return ""


def to_row(item, workflow):
    question = _get_field(item, "Question", "question")
    answer = _get_field(item, "Answer", "answer")
    return {
        "prompt": [{"role": "user", "content": question}],
        "ability": "LocalSearch@bamboogle",
        "extra_info": {
            "query": question,
            "answer": answer,
            "problem_statement": question,
            "type": "multi-hop",
            "level": "unknown",
            "supporting_facts": [],   # Bamboogle has no annotated supporting facts
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--hf_repo", default="chiayewken/bamboogle")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading Bamboogle from HuggingFace: {args.hf_repo} ...")
    ds_dict = None
    try:
        from datasets import load_dataset
        ds = load_dataset(args.hf_repo, cache_dir=args.cache_dir)
        ds_dict = {k: list(v) for k, v in ds.items()}
        print(f"Loaded via datasets.load_dataset, splits: {list(ds_dict.keys())}")
    except Exception as e:
        msg = str(e)
        if ("scripts are no longer supported" in msg or "trust_remote_code" in msg.lower()
                or "loading script" in msg.lower()):
            print(f"datasets refused script-based repo; falling back to HF Hub auto-convert parquet refs ...")
            ds_dict = _hf_parquet_load(args.hf_repo, cache_dir=args.cache_dir)
            print(f"Loaded via parquet refs, splits: {list(ds_dict.keys())}")
        else:
            raise

    # Bamboogle is eval-only. Pick whatever split is available; collapse all into "test".
    items = []
    for split_name, split_items in ds_dict.items():
        items.extend(split_items)
    print(f"Total Bamboogle items: {len(items)}")

    # FoldAgent (search_branch)
    rows = [to_row(t, "search_branch") for t in items]
    pd.DataFrame(rows).to_parquet(f"{args.out_dir}/bamboogle_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/bamboogle_test.parquet  ({len(rows)} rows, search_branch)")

    # ContextGraph (search_graph)
    rows_g = [to_row(t, "search_graph") for t in items]
    pd.DataFrame(rows_g).to_parquet(f"{args.out_dir}/bamboogle_graph_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/bamboogle_graph_test.parquet  ({len(rows_g)} rows, search_graph)")

    print("\nSample questions:")
    for t in items[:5]:
        print(f"  Q: {_get_field(t, 'Question', 'question')}")
        print(f"    A: {_get_field(t, 'Answer', 'answer')}")


if __name__ == "__main__":
    main()
