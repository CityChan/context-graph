#!/usr/bin/env python3
"""Build the 2WikiMultiHopQA search corpus (distractor-pool variant).

Mirrors build_hotpotqa_corpus.py: extracts the per-question distractor
paragraphs from 2WikiMQA (HF default `xanhho/2WikiMultihopQA`), dedupes
by article title, and writes a parquet matching the schema the search
server expects:

  data/2wikimqa_corpus.parquet
    docid : "2wiki_<i>"        stable monotonic id
    url   : Wikipedia URL
    title : article title
    text  : concatenated paragraph sentences

Why distractor-pool: 2WikiMQA's train (~167K) + dev (~12K) × ~10
candidate paragraphs dedupe to ~150-200K unique articles. Every gold
supporting fact is in this pool, so a search server backed by it can
answer every train/val question. Fullwiki is left to a follow-up.

Run on a Vista login node (needs internet for HF download).

Usage:
  python scripts/build_2wikimqa_corpus.py
  python scripts/build_2wikimqa_corpus.py --include_val_only   # smaller pool
  python scripts/build_2wikimqa_corpus.py --hf_repo voidful/2WikiMultihopQA
"""

import os
import argparse

import pandas as pd


def _hf_parquet_load(repo, cache_dir=None):
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


def iter_articles(item):
    """Yield (title, paragraph_text) for every article in a 2WikiMQA distractor row.

    HF parallel-array form: {"title": [t1, ...], "content": [[s1,s2,...], ...]}
                       or  {"title": [t1, ...], "sentences": [[s1,s2,...], ...]}
    Older list-of-dicts form: [{"title", "content" | "sentences"}, ...]
    """
    ctx = item.get("context", {})
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sents_field = ctx.get("content", ctx.get("sentences", []))
        for t, sents in zip(titles, sents_field):
            yield t, " ".join(sents)
    else:
        for entry in ctx:
            t = entry.get("title", "")
            sents = entry.get("content", entry.get("sentences", []))
            yield t, " ".join(sents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/2wikimqa_corpus.parquet")
    parser.add_argument("--hf_repo", default="xanhho/2WikiMultihopQA",
                        help="HF dataset repo (default: xanhho/2WikiMultihopQA).")
    parser.add_argument("--cache_dir", default=None,
                        help="HF datasets cache dir (defaults to $HF_HOME/datasets or ~/.cache/huggingface).")
    parser.add_argument("--include_val_only", action="store_true",
                        help="Skip train split — useful for a smaller smoke-test corpus.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"Loading 2WikiMultiHopQA from HuggingFace: {args.hf_repo} ...")
    ds_dict = None
    try:
        from datasets import load_dataset
        ds = load_dataset(args.hf_repo, cache_dir=args.cache_dir)
        ds_dict = {k: list(v) for k, v in ds.items()}
    except Exception as e:
        msg = str(e)
        if ("scripts are no longer supported" in msg or "trust_remote_code" in msg.lower()
                or "loading script" in msg.lower()):
            print("datasets refused script-based repo; falling back to HF Hub auto-convert parquet refs ...")
            ds_dict = _hf_parquet_load(args.hf_repo, args.cache_dir)
        else:
            raise
    print(f"Splits: { {k: len(v) for k, v in ds_dict.items()} }")

    # title -> paragraph text (last write wins; small whitespace differences OK)
    title_to_text = {}
    val_key = "validation" if "validation" in ds_dict else ("dev" if "dev" in ds_dict else None)
    splits = [val_key] if args.include_val_only else (["train", val_key] if val_key else ["train"])
    splits = [s for s in splits if s and s in ds_dict]
    for split in splits:
        rows = ds_dict[split]
        for i, item in enumerate(rows):
            for title, text in iter_articles(item):
                if not title or not text:
                    continue
                if title not in title_to_text:
                    title_to_text[title] = text
            if (i + 1) % 10000 == 0:
                print(f"  {split}: scanned {i+1}/{len(rows)} rows, {len(title_to_text)} unique articles so far")
        print(f"  {split}: done. {len(title_to_text)} unique articles after this split.")

    print(f"Total unique articles: {len(title_to_text)}")

    rows_out = []
    for i, (title, text) in enumerate(sorted(title_to_text.items())):
        rows_out.append({
            "docid": f"2wiki_{i}",
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "title": title,
            "text": text,
        })

    df = pd.DataFrame(rows_out)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}  ({len(df)} rows)")

    # Quick stats
    lengths = df["text"].str.split().str.len()
    print(f"Paragraph word-count: mean={lengths.mean():.1f}  median={int(lengths.median())}  "
          f"p95={int(lengths.quantile(0.95))}  max={int(lengths.max())}")
    print("\nSample rows:")
    for _, r in df.head(3).iterrows():
        snippet = r["text"][:160].replace("\n", " ")
        print(f"  {r['docid']}  {r['title']!r}\n    {snippet}...")


if __name__ == "__main__":
    main()
