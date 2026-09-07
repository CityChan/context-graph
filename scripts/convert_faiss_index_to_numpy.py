#!/usr/bin/env python3
"""Convert a flat FAISS index to an atomically published float16 NumPy matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def convert(index_path: Path, output_path: Path, chunk_rows: int) -> None:
    import faiss

    index = faiss.read_index(str(index_path))
    if index.ntotal <= 0 or index.d <= 0:
        raise RuntimeError(f"invalid FAISS index shape: ({index.ntotal}, {index.d})")

    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float16, shape=(index.ntotal, index.d)
    )
    for start in range(0, index.ntotal, chunk_rows):
        count = min(chunk_rows, index.ntotal - start)
        matrix[start : start + count] = index.reconstruct_n(start, count)
        matrix.flush()
        print(f"converted_rows={start + count}/{index.ntotal}", flush=True)
    del matrix
    os.replace(partial, output_path)
    loaded = np.load(output_path, mmap_mode="r")
    if loaded.shape != (index.ntotal, index.d) or loaded.dtype != np.float16:
        raise RuntimeError(
            f"converted matrix verification failed: shape={loaded.shape}, dtype={loaded.dtype}"
        )
    print(
        {
            "output": str(output_path),
            "rows": int(loaded.shape[0]),
            "dimensions": int(loaded.shape[1]),
            "dtype": str(loaded.dtype),
            "bytes": output_path.stat().st_size,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.index, args.output, args.chunk_rows)
