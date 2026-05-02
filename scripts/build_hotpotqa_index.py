#!/usr/bin/env python3
"""Encode the HotpotQA corpus with a Qwen3-Embedding model.

Reads `data/hotpotqa_corpus.parquet` (built by build_hotpotqa_corpus.py) and
writes `data/hotpotqa_corpus_embeddings.pkl` in the format
`envs/search_server.py` expects:

  {
    "embeddings": torch.Tensor of shape (N, D), L2-normalized; dtype matches
                  the embedder's compute dtype (default bfloat16) so the
                  search server's torch.mm against BF16 query embeddings
                  doesn't fail with a dtype mismatch.
    "docids":     list[str] of length N (same order as the rows in the parquet),
  }

This pickle + the corpus parquet together fully replace the HuggingFace
dataset+embedding paths the search server defaults to (`Tevatron/browsecomp-...`).

Default embedder is Qwen3-Embedding-4B — chosen so that it can co-locate
on the same GH200 as the 30B trainer/vLLM stack on the main 8-node run
without OOMing. Override with `--model Qwen/Qwen3-Embedding-8B` for fullwiki
or higher-quality runs (only safe when the search server is on a *dedicated*
GPU node).

Single-GPU only — straightforward to run as one sbatch on gh-dev. For ~80-100K
distractor-pool docs at batch_size=32, max_length=512, expected wall time is
~10-20 min.

Usage:
  python scripts/build_hotpotqa_index.py
  python scripts/build_hotpotqa_index.py --model Qwen/Qwen3-Embedding-8B --batch_size 16
"""

import argparse
import os
import pickle
import sys
import time

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool by the last *real* token (mirrors envs/search_server.py)."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/hotpotqa_corpus.parquet")
    parser.add_argument("--out", default="data/hotpotqa_corpus_embeddings.pkl")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B",
                        help="HF id of the embedder. Defaults to 4B for GPU-shared deployments.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512,
                        help="Token cap per doc. HotpotQA paragraphs are short; 512 covers >99%.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Encode only the first N docs (debug).")
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"ERROR: corpus parquet not found at {args.corpus}", file=sys.stderr)
        print("Build it first:  python scripts/build_hotpotqa_corpus.py", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — Qwen3-Embedding encoding is not realistic on CPU.",
              file=sys.stderr)
        sys.exit(2)
    device = torch.device("cuda:0")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"Loading corpus from {args.corpus} ...")
    df = pd.read_parquet(args.corpus)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"  {len(df)} rows")

    docids = df["docid"].tolist()
    # Title prefix gives a strong NER hint to the embedder; matches typical
    # Wikipedia retrieval recipes.
    texts = (df["title"].fillna("") + ". " + df["text"].fillna("")).tolist()

    print(f"Loading embedder: {args.model} ({args.dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    # Use flash-attn if it's compiled in; the existing server requests it
    # explicitly, but here we just let HF pick whatever is available so this
    # script also works on dev environments without flash-attn.

    N = len(texts)
    all_embs = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            batch = texts[start : start + args.batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=args.max_length,
                            return_tensors="pt").to(device)
            out = model(**enc)
            emb = last_token_pool(out.last_hidden_state, enc["attention_mask"])
            # Keep the embedder's compute dtype on save: the search server
            # does torch.mm(query_bf16, corpus.T) and a dtype mismatch errors
            # out. F.normalize returns input dtype.
            emb = F.normalize(emb, p=2, dim=1).cpu()
            all_embs.append(emb)

            if (start // args.batch_size) % 50 == 0:
                done = start + len(batch)
                elapsed = time.time() - t0
                rate = done / max(1e-3, elapsed)
                eta = (N - done) / max(1e-3, rate)
                print(f"  {done}/{N}  ({rate:.1f} docs/s, ETA {eta/60:.1f} min)", flush=True)

    embeddings = torch.cat(all_embs, dim=0)
    assert embeddings.shape[0] == N, f"shape mismatch: {embeddings.shape[0]} vs {N}"
    print(f"Encoded {N} docs into shape {tuple(embeddings.shape)} in {time.time() - t0:.0f}s")

    print(f"Writing {args.out} ...")
    with open(args.out, "wb") as f:
        pickle.dump({"embeddings": embeddings, "docids": docids}, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Wrote {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
