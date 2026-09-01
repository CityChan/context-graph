#!/usr/bin/env python3
"""Check that a merged Hugging Face model differs from its exported base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--merged-model", required=True)
    parser.add_argument("--tensor-suffix", default="q_proj.weight")
    return parser.parse_args()


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid weight map: {index_path}")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_model)
    merged_dir = Path(args.merged_model)
    base_map = load_weight_map(base_dir)
    merged_map = load_weight_map(merged_dir)
    tensor_name = next(
        (
            name
            for name in base_map
            if name.endswith(args.tensor_suffix) and name in merged_map
        ),
        None,
    )
    if tensor_name is None:
        raise ValueError(
            f"no common tensor ending with {args.tensor_suffix!r} was found"
        )

    from safetensors import safe_open

    with safe_open(
        str(base_dir / base_map[tensor_name]), framework="pt", device="cpu"
    ) as base_file:
        base_tensor = base_file.get_tensor(tensor_name).float()
    with safe_open(
        str(merged_dir / merged_map[tensor_name]), framework="pt", device="cpu"
    ) as merged_file:
        merged_tensor = merged_file.get_tensor(tensor_name).float()
    if base_tensor.shape != merged_tensor.shape:
        raise ValueError(
            f"tensor shape mismatch: {tuple(base_tensor.shape)} != {tuple(merged_tensor.shape)}"
        )

    delta = merged_tensor - base_tensor
    result = {
        "tensor": tensor_name,
        "shape": list(delta.shape),
        "mean_abs_delta": delta.abs().mean().item(),
        "max_abs_delta": delta.abs().max().item(),
        "nonzero": int(delta.count_nonzero().item()),
        "merge_changed_tensor": bool(delta.count_nonzero().item()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
