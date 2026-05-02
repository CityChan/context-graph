#!/usr/bin/env python3
"""Build the HotpotQA search corpus (distractor-pool variant).

For Stage 2 of the HotpotQA wire-up we need a local document collection that
the agent can search/open against. This script extracts the per-question
distractor paragraphs from HotpotQA (HF `hotpot_qa` repo, `distractor` config),
dedupes them by article title, and writes a parquet file with the schema the
search server expects:

  data/hotpotqa_corpus.parquet
    docid  : "hotpot_<i>"      stable monotonic id
    url    : Wikipedia URL    (https://en.wikipedia.org/wiki/<Title>)
    title  : article title
    text   : concatenated paragraph sentences

Why distractor-pool instead of fullwiki: the distractor split is ~97K rows
(train + val) × 10 candidate paragraphs, which dedupes to ~80-100K unique
articles. Every gold supporting fact is guaranteed to be in this pool, so a
search server backed by it can answer every train/val question. Fullwiki
(~5M paragraphs) is the realistic setting and is left to a follow-up
(`build_hotpotqa_fullwiki_corpus.py`).

This script needs internet to pull from HuggingFace — run on a Vista login
node, not on a compute node. Once `data/hotpotqa_corpus.parquet` exists and
the HF cache is populated, the training nodes read offline.

Usage:
  python scripts/build_hotpotqa_corpus.py
  python scripts/build_hotpotqa_corpus.py --include_val_only   # smaller pool
"""

import os
import argparse

import pandas as pd


def iter_articles(item):
    """Yield (title, paragraph_text) for every article in a HotpotQA distractor row.

    HF schema returns `context` as a parallel-array dict:
      {"title": [t1, t2, ...], "sentences": [[s1,s2,...], [s1,s2,...], ...]}
    Older snapshots may give a list of {"title", "sentences"} dicts — handle both.
    """
    ctx = item.get("context", {})
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        for t, sents in zip(titles, sentences):
            yield t, " ".join(sents)
    else:
        for entry in ctx:
            t = entry.get("title", "")
            sents = entry.get("sentences", [])
            yield t, " ".join(sents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/hotpotqa_corpus.parquet")
    parser.add_argument("--cache_dir", default=None,
                        help="HF datasets cache dir (defaults to $HF_HOME/datasets or ~/.cache/huggingface).")
    parser.add_argument("--include_val_only", action="store_true",
                        help="Skip train split — useful for a smaller smoke-test corpus.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("`datasets` not installed. `pip install datasets` (should be in cxtgraph env).") from e

    print("Loading HotpotQA (distractor config) from HuggingFace...")
    ds = load_dataset("hotpot_qa", "distractor", trust_remote_code=True, cache_dir=args.cache_dir)
    print(f"Splits: { {k: len(v) for k, v in ds.items()} }")

    # title -> paragraph text (last write wins; HotpotQA articles are usually
    # consistent across rows, but small whitespace differences are harmless).
    title_to_text = {}
    splits = ["validation"] if args.include_val_only else ["train", "validation"]
    for split in splits:
        if split not in ds:
            continue
        rows = ds[split]
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
            "docid": f"hotpot_{i}",
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
