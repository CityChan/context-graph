#!/usr/bin/env python3
"""Fail fast when the active Transformers build cannot load a model config."""

from __future__ import annotations

import argparse
import sys

import transformers
from transformers import AutoConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()

    print(f"transformers: {transformers.__version__}")
    try:
        config = AutoConfig.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        print(
            "ERROR: active Transformers cannot load "
            f"{args.model_path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            "Use a separate Qwen3.5-compatible environment and rerun with "
            "CONDA_ENV_NAME set to that environment.",
            file=sys.stderr,
        )
        return 2

    print(
        "model config: "
        f"class={type(config).__name__} "
        f"model_type={getattr(config, 'model_type', None)} "
        f"architectures={getattr(config, 'architectures', None)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
