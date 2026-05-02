#!/usr/bin/env python3
"""BM25 search server over the HotpotQA distractor-pool corpus.

A lightweight retrieval backend that loads `data/hotpotqa_corpus.parquet`
(built by build_hotpotqa_corpus.py) and serves the same API contract as
`multihop_search_server.py` and `envs/search_server.py`:

  POST /search  {"query": str, "k": int}
    -> {"results": [{"docid", "url", "text"}, ...]}
  POST /open    {"docid": str}  or  {"url": str}
    -> {"results": [{"docid", "url", "text"}]}

Why exists alongside envs/search_server.py:
  - The Qwen3-Embedding-4B server needs ~10 GB of GPU memory; on the 4B
    1-node smoke that GPU is shared with the trainer's vLLM. If the goal
    is just to validate the ContextGraph code path (search_graph workflow,
    process_reward=graph fires, etc.), BM25 removes the GPU contention
    entirely and lets the smoke iterate ~3x faster.
  - HotpotQA is named-entity-heavy, so BM25 R@10 is ~60-70% — weak vs.
    Qwen3-Embedding's ~85-90% but enough for the reward signal to fire
    on simpler bridge questions.

Implementation:
  - Inverted index: term -> [(doc_idx, term_freq), ...] so per-query work
    is O(sum_t |postings(t)|) instead of O(N * |query|). Sub-100ms /search
    on a single core for ~100K docs.
  - Standard Okapi BM25 (k1=1.5, b=0.75). No stemming, lowercase \\w+
    tokens. Title is repeated 2x to give NER queries a boost.

Default port: 18999 (matches the synthetic multihop server) so the
training scripts' LOCAL_SEARCH_URL contract is identical regardless of
which retrieval backend is used.
"""

import argparse
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


# ── Globals populated at startup ──
DOCS = []                  # list of dict: {docid, url, title, text}
DOCS_BY_ID = {}            # docid -> doc index
DOCS_BY_URL = {}           # url   -> doc index
DOC_LEN = []               # token count per doc
AVGDL = 0.0                # average doc length
N = 0                      # number of docs
IDF = {}                   # term -> idf
POSTINGS = defaultdict(list)   # term -> [(doc_idx, tf_in_doc), ...]

K1 = 1.5
B = 0.75


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def build_index(parquet_path: str):
    global DOCS, DOCS_BY_ID, DOCS_BY_URL, DOC_LEN, AVGDL, N, IDF, POSTINGS

    if not os.path.exists(parquet_path):
        print(f"ERROR: corpus parquet not found at {parquet_path}", file=sys.stderr)
        print("Build it first on a login node:", file=sys.stderr)
        print("  python scripts/build_hotpotqa_corpus.py", file=sys.stderr)
        sys.exit(2)

    print(f"Loading corpus from {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df)} rows")

    DOCS = df.to_dict(orient="records")
    DOCS_BY_ID = {d["docid"]: i for i, d in enumerate(DOCS)}
    DOCS_BY_URL = {d["url"]: i for i, d in enumerate(DOCS)}

    print("Tokenizing + building inverted index ...")
    df_count = Counter()
    DOC_LEN = [0] * len(DOCS)
    POSTINGS = defaultdict(list)

    for i, doc in enumerate(DOCS):
        # 2x title boost: cheap and helps NER-style queries a lot.
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        tokens = tokenize(title + " " + title + " " + text)
        DOC_LEN[i] = len(tokens)
        tf = Counter(tokens)
        for term, freq in tf.items():
            POSTINGS[term].append((i, freq))
            df_count[term] += 1

    N = len(DOCS)
    AVGDL = (sum(DOC_LEN) / N) if N else 1.0

    # Okapi BM25 IDF (smoothed). Floor at a small positive value so very
    # common terms can't drive scores negative.
    IDF = {
        term: max(1e-6, math.log((N - df + 0.5) / (df + 0.5) + 1.0))
        for term, df in df_count.items()
    }

    print(f"  vocab: {len(IDF)} terms")
    print(f"  avgdl: {AVGDL:.1f} tokens")
    print(f"  posting list size sum: {sum(len(v) for v in POSTINGS.values())}")


def bm25_search(query: str, k: int):
    q_tokens = [t for t in tokenize(query) if t in IDF]
    if not q_tokens:
        # Stopword-only query (rare); just return some docs to keep the
        # agent moving instead of 500ing.
        return [DOCS[i] for i in range(min(k, N))]

    scores = defaultdict(float)
    q_term_counts = Counter(q_tokens)
    for term, q_count in q_term_counts.items():
        idf = IDF[term]
        for doc_idx, tf in POSTINGS.get(term, ()):
            dl = DOC_LEN[doc_idx]
            denom = tf + K1 * (1.0 - B + B * dl / AVGDL)
            scores[doc_idx] += idf * (tf * (K1 + 1.0)) / denom * q_count

    if not scores:
        return [DOCS[i] for i in range(min(k, N))]

    top = sorted(scores.items(), key=lambda x: -x[1])[:k]
    return [DOCS[i] for i, _ in top]


# ── HTTP layer ──

class QueryRequest(BaseModel):
    query: str
    k: int = 5


class OpenRequest(BaseModel):
    docid: Optional[str] = None
    url: Optional[str] = None


def _strip(d):
    return {"docid": d["docid"], "url": d["url"], "text": d["text"]}


@app.post("/search")
async def search(req: QueryRequest):
    hits = bm25_search(req.query, max(1, int(req.k)))
    return {"results": [_strip(d) for d in hits]}


@app.post("/open")
async def open_page(req: OpenRequest):
    idx = None
    if req.docid and req.docid in DOCS_BY_ID:
        idx = DOCS_BY_ID[req.docid]
    elif req.url and req.url in DOCS_BY_URL:
        idx = DOCS_BY_URL[req.url]
    if idx is None:
        # Match the synthetic multihop server's behavior: fallback to a doc
        # rather than 404, so the agent gets *something* back.
        idx = 0
    return {"results": [_strip(DOCS[idx])]}


@app.get("/health")
async def health():
    return {"status": "healthy", "corpus_size": N, "vocab_size": len(IDF)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/hotpotqa_corpus.parquet")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18999)
    args = parser.parse_args()

    build_index(args.corpus)
    print(f"HotpotQA BM25 server: {N} docs indexed. Starting on {args.host}:{args.port} ...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
