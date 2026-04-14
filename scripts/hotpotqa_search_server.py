#!/usr/bin/env python3
"""HotpotQA context-based search server.

Unlike mock_search_server.py (which returns pre-canned irrelevant results),
this server builds a search index from HotpotQA's own context paragraphs.
Each HotpotQA sample comes with 10 Wikipedia paragraphs (2 gold + 8 distractors).
We index ALL of them so the agent can actually find useful information via search.

This makes HotpotQA a meaningful benchmark:
  - Agent must search for the right paragraphs among ~5000 indexed passages
  - 2-hop questions require finding info from TWO different paragraphs
  - ContextGraph's cross-branch reasoning becomes valuable (connect entity A + B)

Usage:
    python scripts/hotpotqa_search_server.py [--data_dir data] [--port 18999]

The server loads hotpotqa_train.parquet and hotpotqa_test.parquet, extracts all
context paragraphs, builds a simple TF-IDF index, and serves /search and /open.
"""

import os
import re
import argparse
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from collections import Counter
import math


app = FastAPI()

# Global index, populated at startup
DOCS = []       # list of {"docid": str, "title": str, "text": str}
DOCS_BY_ID = {} # docid -> doc dict
IDF = {}        # word -> idf score
DOC_TFIDF = []  # list of {word: tfidf_score} per doc


class QueryRequest(BaseModel):
    query: str
    k: int = 10


class OpenRequest(BaseModel):
    docid: Optional[str] = None
    url: Optional[str] = None


def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r'\w+', text.lower())


def build_index(data_dir: str = "data"):
    """Load HotpotQA parquets and index all context paragraphs."""
    import pandas as pd
    from datasets import load_dataset

    global DOCS, DOCS_BY_ID, IDF, DOC_TFIDF

    print("Loading HotpotQA dataset for indexing...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor")

    # Collect paragraphs from both train and validation
    seen_titles = set()
    for split_name in ["train", "validation"]:
        split = ds[split_name]
        # Only index first 2000 samples to keep memory reasonable
        for i, item in enumerate(split):
            if i >= 2000:
                break
            context = item.get("context", {})
            titles = context.get("title", [])
            sentences_list = context.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                text = " ".join(sentences)
                docid = f"hotpot_{len(DOCS)}"
                doc = {
                    "docid": docid,
                    "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    "title": title,
                    "text": f"{title}\n\n{text}",
                }
                DOCS.append(doc)
                DOCS_BY_ID[docid] = doc

    print(f"Indexed {len(DOCS)} unique paragraphs from HotpotQA")

    # Build TF-IDF index
    N = len(DOCS)
    # Document frequency
    df = Counter()
    doc_tokens = []
    for doc in DOCS:
        tokens = tokenize(doc["text"])
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] += 1
        doc_tokens.append(tokens)

    # IDF
    IDF = {word: math.log(N / (count + 1)) for word, count in df.items()}

    # TF-IDF per document
    for tokens in doc_tokens:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        tfidf = {}
        for word, count in tf.items():
            tfidf[word] = (count / total) * IDF.get(word, 0)
        DOC_TFIDF.append(tfidf)

    print(f"TF-IDF index built. Vocabulary size: {len(IDF)}")


MAX_DOC_TOKENS = 200  # Cap each returned paragraph to ~200 words


def truncate_doc(text: str, max_tokens: int = MAX_DOC_TOKENS) -> str:
    """Truncate document text to max_tokens words, preserving sentence boundaries."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    # Find last period within limit
    truncated = " ".join(words[:max_tokens])
    last_period = truncated.rfind(".")
    if last_period > len(truncated) // 2:
        return truncated[:last_period + 1]
    return truncated + "..."


def search_tfidf(query: str, k: int = 10) -> list[dict]:
    """Score documents against query using TF-IDF cosine similarity."""
    query_tokens = tokenize(query)
    query_tf = Counter(query_tokens)
    total = len(query_tokens) if query_tokens else 1

    query_tfidf = {}
    for word, count in query_tf.items():
        if word in IDF:
            query_tfidf[word] = (count / total) * IDF[word]

    if not query_tfidf:
        return [{"docid": d["docid"], "url": d["url"],
                 "text": truncate_doc(d["text"])} for d in DOCS[:k]]

    scores = []
    for i, doc_tf in enumerate(DOC_TFIDF):
        score = sum(query_tfidf.get(w, 0) * doc_tf.get(w, 0) for w in query_tfidf)
        if score > 0:
            scores.append((score, i))

    scores.sort(key=lambda x: -x[0])
    results = []
    for score, idx in scores[:k]:
        doc = DOCS[idx]
        results.append({
            "docid": doc["docid"],
            "url": doc["url"],
            "text": truncate_doc(doc["text"]),
        })

    if not results:
        results = [{"docid": d["docid"], "url": d["url"],
                     "text": truncate_doc(d["text"])} for d in DOCS[:k]]

    return results


@app.post("/search")
async def search(req: QueryRequest):
    results = search_tfidf(req.query, req.k)
    return {"results": results}


@app.post("/open")
async def open_page(req: OpenRequest):
    if req.docid and req.docid in DOCS_BY_ID:
        doc = DOCS_BY_ID[req.docid]
        return {"results": [{"docid": doc["docid"], "url": doc["url"], "text": doc["text"]}]}
    if req.url:
        for doc in DOCS:
            if doc["url"] == req.url:
                return {"results": [{"docid": doc["docid"], "url": doc["url"], "text": doc["text"]}]}
    # Fallback
    doc = DOCS[0]
    return {"results": [{"docid": doc["docid"], "url": doc["url"], "text": doc["text"]}]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18999)
    args = parser.parse_args()

    build_index()
    print(f"Starting HotpotQA search server on port {args.port}...")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
