#!/usr/bin/env python3
"""Embed a local search corpus for envs/search_server.py."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def last_token_pool(last_hidden_states, attention_mask):
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive")

    import pandas as pd
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)
    frame = pd.read_parquet(input_path)
    missing = sorted({"docid", "text"} - set(frame.columns))
    if missing:
        raise ValueError(f"corpus is missing required columns: {missing}")
    docids = frame["docid"].astype(str).tolist()
    if not docids or len(docids) != len(set(docids)):
        raise ValueError("corpus docids must be non-empty and unique")
    texts = frame["text"].fillna("").astype(str).tolist()
    if "title" in frame.columns:
        titles = frame["title"].fillna("").astype(str).tolist()
        texts = [f"{title}. {text}" if title.strip() else text for title, text in zip(titles, texts)]
    if any(not text.strip() for text in texts):
        raise ValueError("corpus contains empty text")

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    batches = []
    for start in tqdm(range(0, len(texts), args.batch_size), desc="Embed corpus"):
        batch = tokenizer(
            texts[start : start + args.batch_size],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.inference_mode():
            outputs = model(**batch)
            embeddings = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            embeddings = F.normalize(embeddings.float(), p=2, dim=1).to(torch.bfloat16).cpu()
        batches.append(embeddings)

    corpus_embeddings = torch.cat(batches, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(
            {"embeddings": corpus_embeddings, "docids": docids},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    temporary_path.replace(output_path)
    manifest = {
        "schema_version": "contextgraph.local_corpus_embeddings.v1",
        "input": str(input_path),
        "output": str(output_path),
        "model": args.model,
        "rows": len(docids),
        "embedding_shape": list(corpus_embeddings.shape),
        "embedding_dtype": str(corpus_embeddings.dtype),
        "max_length": args.max_length,
        "normalized": True,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
