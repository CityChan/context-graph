#!/usr/bin/env python3
"""Validate and tokenize a ContextGraph multi-turn SFT parquet sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--loss-mask-mode", choices=("per_message", "assistant_tokens", "chatml"), default="assistant_tokens"
    )
    parser.add_argument("--all-samples", action="store_true", help="Tokenize and validate every parquet row")
    return parser.parse_args()


def stage(message: str) -> None:
    print(f"SFT data preflight stage: {message}", flush=True)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file() or data_path.stat().st_size == 0:
        raise SystemExit(f"missing or empty SFT parquet: {data_path}")

    stage("import pandas and OmegaConf")
    import pandas as pd
    from omegaconf import OmegaConf

    stage("import VERL tokenizer and multi-turn dataset")
    from verl.utils import hf_tokenizer
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    stage("read parquet")
    frame = pd.read_parquet(data_path)
    required_columns = {"messages", "tools", "enable_thinking"}
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        raise SystemExit(f"SFT parquet is missing columns: {missing}")
    if frame.empty:
        raise SystemExit("SFT parquet has no rows")

    stage("load tokenizer")
    tokenizer = hf_tokenizer(args.tokenizer, trust_remote_code=True, local_files_only=True)
    config = OmegaConf.create(
        {
            "messages_key": "messages",
            "tools_key": "tools",
            "enable_thinking_key": "enable_thinking",
            "max_length": args.max_length,
            "truncation": "right",
            "pad_mode": "right",
            "loss_mask_mode": args.loss_mask_mode,
        }
    )
    selected_samples = -1 if args.all_samples else 1
    stage("construct full multi-turn dataset" if args.all_samples else "construct one-row multi-turn dataset")
    dataset = MultiTurnSFTDataset(
        parquet_files=str(data_path),
        tokenizer=tokenizer,
        config=config,
        max_samples=selected_samples,
    )
    stage("tokenize all samples" if args.all_samples else "tokenize first sample")
    input_token_counts = []
    loss_token_counts = []
    for index in range(len(dataset)):
        try:
            sample = dataset[index]
        except Exception as exc:
            raise RuntimeError(f"failed to tokenize SFT row {index}") from exc
        input_tokens = int(sample["attention_mask"].sum().item())
        loss_tokens = int(sample["loss_mask"].sum().item())
        if input_tokens <= 0:
            raise SystemExit(f"tokenized SFT row {index} has no input tokens")
        if loss_tokens <= 0:
            raise SystemExit(f"tokenized SFT row {index} has no assistant loss tokens")
        input_token_counts.append(input_tokens)
        loss_token_counts.append(loss_tokens)
        if args.all_samples and (index + 1) % 100 == 0:
            print(f"SFT data preflight progress: {index + 1}/{len(dataset)}", flush=True)

    stage("complete")
    print(
        json.dumps(
            {
                "data": str(data_path),
                "rows": len(frame),
                "tokenizer": args.tokenizer,
                "max_length": args.max_length,
                "loss_mask_mode": args.loss_mask_mode,
                "samples_checked": len(input_token_counts),
                "sample_input_tokens": input_token_counts[0],
                "sample_loss_tokens": loss_token_counts[0],
                "max_input_tokens": max(input_token_counts),
                "min_loss_tokens": min(loss_token_counts),
                "max_loss_tokens": max(loss_token_counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
