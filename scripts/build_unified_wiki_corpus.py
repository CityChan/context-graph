#!/usr/bin/env python3
"""Build a unified Wikipedia corpus that covers both HotpotQA and 2WikiMQA.

Reads both HF datasets directly (HotpotQA distractor + 2WikiMQA distractor),
extracts every article's title + paragraph text, dedupes by title, and writes
a single parquet that the search server can serve to BOTH training pipelines:

  data/wiki_corpus.parquet
    docid : "wiki_<i>"           stable monotonic id
    url   : Wikipedia URL
    title : article title
    text  : concatenated paragraph sentences

Why one corpus: the two distractor pools overlap heavily (both are
Wikipedia paragraphs from the same era of dumps) and using a single
corpus eliminates "retrieval quality differs across datasets" as a
confounder when comparing ContextGraph vs FoldAgent results across
HotpotQA and 2WikiMQA. Expected size: ~200-250K unique articles.

Run on a Vista login node (needs internet for HF download). Once
data/wiki_corpus.parquet exists and the HF cache is populated, training
nodes read offline.

Usage:
  python scripts/build_unified_wiki_corpus.py
  python scripts/build_unified_wiki_corpus.py --2wikimqa_repo voidful/2WikiMultihopQA
"""

import argparse
import os

import pandas as pd


def _hf_parquet_load(repo, config=None, cache_dir=None):
    """Load HF dataset via auto-converted parquet refs (bypass loading scripts).

    For repos with multiple configs (e.g. hotpot_qa has 'distractor', 'fullwiki'),
    pass `config` to filter; the parquet refs path format is
    <config>/<split>/<num>.parquet so we filter by parts[0].
    """
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


def _load_with_fallback(repo, config=None, cache_dir=None):
    """Try datasets.load_dataset; on script-rejection fall back to parquet refs."""
    try:
        from datasets import load_dataset
        if config is not None:
            ds = load_dataset(repo, config, cache_dir=cache_dir)
        else:
            ds = load_dataset(repo, cache_dir=cache_dir)
        return {k: list(v) for k, v in ds.items()}
    except Exception as e:
        msg = str(e)
        if ("scripts are no longer supported" in msg or "trust_remote_code" in msg.lower()
                or "loading script" in msg.lower()):
            print(f"  datasets refused script-based repo {repo}; falling back to parquet refs ...")
            return _hf_parquet_load(repo, config=config, cache_dir=cache_dir)
        raise


def iter_articles_hotpot(item):
    """Yield (title, paragraph_text) for HotpotQA distractor row."""
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


def iter_articles_2wiki(item):
    """Yield (title, paragraph_text) for 2WikiMQA distractor row.

    2WikiMQA mirrors expose context with either `content` or `sentences` field,
    so accept both.
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
    parser.add_argument("--out", default="data/wiki_corpus.parquet")
    parser.add_argument("--hotpot_repo", default="hotpot_qa")
    parser.add_argument("--hotpot_config", default="distractor")
    parser.add_argument("--two_wiki_repo", dest="two_wiki_repo", default="xanhho/2WikiMultihopQA")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--include_val_only", action="store_true",
                        help="Skip train splits — useful for a smaller smoke-test corpus.")
    parser.add_argument("--skip_hotpot", action="store_true",
                        help="Skip HotpotQA (e.g. when corpus is too big and you only want 2WikiMQA).")
    parser.add_argument("--skip_2wiki", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    title_to_text = {}

    if not args.skip_hotpot:
        print(f"Loading HotpotQA ({args.hotpot_repo}, config={args.hotpot_config}) ...")
        ds_hp = _load_with_fallback(args.hotpot_repo, config=args.hotpot_config, cache_dir=args.cache_dir)
        print(f"  HotpotQA splits: { {k: len(v) for k, v in ds_hp.items()} }")
        hp_splits = ["validation"] if args.include_val_only else ["train", "validation"]
        for split in hp_splits:
            if split not in ds_hp:
                continue
            rows = ds_hp[split]
            for i, item in enumerate(rows):
                for title, text in iter_articles_hotpot(item):
                    if title and text and title not in title_to_text:
                        title_to_text[title] = text
                if (i + 1) % 10000 == 0:
                    print(f"  hotpot/{split}: {i+1}/{len(rows)} rows scanned, {len(title_to_text)} unique titles")
            print(f"  hotpot/{split}: done. unique titles so far = {len(title_to_text)}")

    if not args.skip_2wiki:
        print(f"Loading 2WikiMQA ({args.two_wiki_repo}) ...")
        ds_2w = _load_with_fallback(args.two_wiki_repo, cache_dir=args.cache_dir)
        print(f"  2WikiMQA splits: { {k: len(v) for k, v in ds_2w.items()} }")
        val_key = "validation" if "validation" in ds_2w else ("dev" if "dev" in ds_2w else None)
        two_wiki_splits = [val_key] if args.include_val_only else (["train", val_key] if val_key else ["train"])
        two_wiki_splits = [s for s in two_wiki_splits if s and s in ds_2w]
        for split in two_wiki_splits:
            rows = ds_2w[split]
            for i, item in enumerate(rows):
                for title, text in iter_articles_2wiki(item):
                    if title and text and title not in title_to_text:
                        title_to_text[title] = text
                if (i + 1) % 10000 == 0:
                    print(f"  2wiki/{split}: {i+1}/{len(rows)} rows scanned, {len(title_to_text)} unique titles")
            print(f"  2wiki/{split}: done. unique titles so far = {len(title_to_text)}")

    print(f"Total unique articles across both datasets: {len(title_to_text)}")

    rows_out = []
    for i, (title, text) in enumerate(sorted(title_to_text.items())):
        rows_out.append({
            "docid": f"wiki_{i}",
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
            "title": title,
            "text": text,
        })

    df = pd.DataFrame(rows_out)
    df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}  ({len(df)} rows)")

    lengths = df["text"].str.split().str.len()
    print(f"Paragraph word-count: mean={lengths.mean():.1f}  median={int(lengths.median())}  "
          f"p95={int(lengths.quantile(0.95))}  max={int(lengths.max())}")
    print("\nSample rows:")
    for _, r in df.head(3).iterrows():
        snippet = r["text"][:160].replace("\n", " ")
        print(f"  {r['docid']}  {r['title']!r}\n    {snippet}...")


if __name__ == "__main__":
    main()
