#!/usr/bin/env python3
"""Generate Musique parquet for FoldAgent / ContextGraph training and eval.

Downloads MuSiQue from HuggingFace (default `dgslibisey/MuSiQue`, "answerable"
config). Schema mirrors make_hotpotqa_data.py / make_2wikimqa_data.py so the
trainer entry points are reusable. Outputs four parquet files:

  data/musique_train.parquet        (workflow=search_branch, FoldAgent)
  data/musique_test.parquet
  data/musique_graph_train.parquet  (workflow=search_graph,  ContextGraph)
  data/musique_graph_test.parquet

MuSiQue's distinguishing feature: each example carries a
`question_decomposition` field — explicit sub-questions and intermediate
answers showing the reasoning chain. We persist it through extra_info so
graph-aware reward shaping can use it later.

Usage:
  python scripts/make_musique_data.py --n_train 5000 --n_val 500
  python scripts/make_musique_data.py --all
  python scripts/make_musique_data.py --hf_repo dgslibisey/MuSiQue --hf_config answerable
"""

import argparse
import json
import os
import random
from collections import Counter

import pandas as pd


def _maybe_json_loads(x):
    """Decode JSON-encoded string if needed; else return as-is."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (json.JSONDecodeError, ValueError):
            return None
    return x


def _hf_parquet_load(repo, config=None, cache_dir=None):
    """Load HF dataset via auto-converted parquet refs (bypasses loading scripts)."""
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
        raise RuntimeError(f"No parquet files at refs/convert/parquet for {repo} (config={config})")
    result = {}
    for split, paths in splits.items():
        dfs = []
        for p in sorted(paths):
            local = hf_hub_download(repo, p, repo_type="dataset",
                                    revision="refs/convert/parquet", cache_dir=cache_dir)
            dfs.append(pd.read_parquet(local))
        result[split] = pd.concat(dfs, ignore_index=True).to_dict("records")
    return result


def _normalize_supporting_facts(paragraphs):
    """Extract supporting_facts from MuSiQue paragraphs (list of dict with is_supporting flag).

    Returns list of [title, paragraph_idx_str] for paragraphs marked is_supporting=True.
    """
    paragraphs = _maybe_json_loads(paragraphs)
    if paragraphs is None:
        return []
    out = []
    if isinstance(paragraphs, list):
        for p in paragraphs:
            if isinstance(p, dict) and p.get("is_supporting"):
                title = p.get("title", "")
                idx = p.get("idx", 0)
                out.append([title, str(idx)])
    return out


def _normalize_decomposition(decomp):
    """Return list of {sub_question, answer, paragraph_support_idx} dicts.

    MuSiQue's question_decomposition is a list of step dicts.
    """
    decomp = _maybe_json_loads(decomp)
    if decomp is None:
        return []
    out = []
    if isinstance(decomp, list):
        for step in decomp:
            if isinstance(step, dict):
                out.append({
                    "question": str(step.get("question", "")),
                    "answer": str(step.get("answer", "")),
                    "paragraph_support_idx": int(step.get("paragraph_support_idx", -1) or -1),
                })
    return out


def to_row(item, workflow):
    answer = item.get("answer", "")
    aliases = item.get("answer_aliases", []) or []
    if isinstance(aliases, str):
        aliases = _maybe_json_loads(aliases) or []

    return {
        "prompt": [{"role": "user", "content": item["question"]}],
        "ability": "LocalSearch@musique",
        "extra_info": {
            "query": item["question"],
            "answer": answer,
            "answer_aliases": list(aliases),
            "problem_statement": item["question"],
            "type": "compositional",  # MuSiQue is all compositional 2-4 hop
            "level": "unknown",
            "supporting_facts": _normalize_supporting_facts(item.get("paragraphs")),
            "decomposition": _normalize_decomposition(item.get("question_decomposition")),
            "workflow": workflow,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--all", action="store_true",
                        help="Use the full MuSiQue answerable train (~19K) + dev (~2.4K).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--hf_repo", default="dgslibisey/MuSiQue")
    parser.add_argument("--hf_config", default="answerable",
                        help="MuSiQue config; 'answerable' is the standard QA subset.")
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading MuSiQue from HuggingFace: {args.hf_repo} (config={args.hf_config}) ...")
    ds_dict = None
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset(args.hf_repo, args.hf_config, cache_dir=args.cache_dir)
        except Exception:
            # Some MuSiQue mirrors don't expose the config; fall through.
            ds = load_dataset(args.hf_repo, cache_dir=args.cache_dir)
        ds_dict = {k: list(v) for k, v in ds.items()}
        print(f"Loaded via datasets.load_dataset, splits: {list(ds_dict.keys())}")
    except Exception as e:
        msg = str(e)
        if ("scripts are no longer supported" in msg or "trust_remote_code" in msg.lower()
                or "loading script" in msg.lower()):
            print(f"datasets refused script-based repo; falling back to HF Hub auto-convert parquet refs ...")
            ds_dict = _hf_parquet_load(args.hf_repo, config=args.hf_config, cache_dir=args.cache_dir)
            print(f"Loaded via parquet refs, splits: {list(ds_dict.keys())}")
        else:
            raise

    # Find train + val splits (typical: train / validation; some mirrors use dev)
    train_key = "train" if "train" in ds_dict else None
    val_key = "validation" if "validation" in ds_dict else ("dev" if "dev" in ds_dict else None)
    if train_key is None or val_key is None:
        raise SystemExit(f"Expected train + validation splits, got {list(ds_dict.keys())}")
    print(f"  {train_key}: {len(ds_dict[train_key])}  {val_key}: {len(ds_dict[val_key])}")

    train_items = list(ds_dict[train_key])
    val_items = list(ds_dict[val_key])

    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    if not args.all:
        train_items = train_items[: args.n_train]
        val_items = val_items[: args.n_val]
    print(f"Using train={len(train_items)}  val={len(val_items)}")

    # FoldAgent (search_branch)
    train_rows = [to_row(t, "search_branch") for t in train_items]
    val_rows = [to_row(t, "search_branch") for t in val_items]
    pd.DataFrame(train_rows).to_parquet(f"{args.out_dir}/musique_train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(f"{args.out_dir}/musique_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/musique_train.parquet  ({len(train_rows)} rows, search_branch)")
    print(f"Wrote {args.out_dir}/musique_test.parquet   ({len(val_rows)} rows, search_branch)")

    # ContextGraph (search_graph)
    train_rows_g = [to_row(t, "search_graph") for t in train_items]
    val_rows_g = [to_row(t, "search_graph") for t in val_items]
    pd.DataFrame(train_rows_g).to_parquet(f"{args.out_dir}/musique_graph_train.parquet", index=False)
    pd.DataFrame(val_rows_g).to_parquet(f"{args.out_dir}/musique_graph_test.parquet", index=False)
    print(f"Wrote {args.out_dir}/musique_graph_train.parquet  ({len(train_rows_g)} rows, search_graph)")
    print(f"Wrote {args.out_dir}/musique_graph_test.parquet   ({len(val_rows_g)} rows, search_graph)")

    print("\nSample questions:")
    for t in train_items[:5]:
        print(f"  Q: {t['question']}")
        print(f"    A: {t.get('answer', '')}")
        decomp = _normalize_decomposition(t.get("question_decomposition"))
        if decomp:
            for i, step in enumerate(decomp[:3]):
                print(f"    step {i+1}: {step['question']} -> {step['answer']}")


if __name__ == "__main__":
    main()
