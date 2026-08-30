#!/usr/bin/env python3
"""Copy a legacy Qwen3.5 LoRA adapter while repairing its text-layer keys."""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path


def load_key_normalizer():
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "lora_adapter.py"
    spec = importlib.util.spec_from_file_location("contextgraph_lora_adapter", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LoRA key normalizer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize_lora_adapter_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from safetensors.torch import load_file, save_file

    normalize_lora_adapter_key = load_key_normalizer()
    source = args.source.resolve()
    target = args.target.resolve()
    config_path = source / "adapter_config.json"
    weights_path = source / "adapter_model.safetensors"

    for path in (config_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing adapter file: {path}")
    if target.exists():
        raise FileExistsError(f"Target already exists; choose a new path: {target}")

    state_dict = load_file(str(weights_path), device="cpu")
    repaired = {}
    changed = 0
    for name, tensor in state_dict.items():
        repaired_name = normalize_lora_adapter_key(name, model_type="qwen3_5")
        if repaired_name in repaired:
            raise ValueError(f"Duplicate LoRA key after normalization: {repaired_name}")
        repaired[repaired_name] = tensor
        changed += repaired_name != name

    legacy_prefix = "base_model.model.model.layers."
    if changed == 0:
        raise ValueError(f"No legacy Qwen3.5 LoRA keys beginning with {legacy_prefix!r} were found")
    if any(name.startswith(legacy_prefix) for name in repaired):
        raise AssertionError("Legacy Qwen3.5 LoRA keys remain after repair")

    target.mkdir(parents=True)
    try:
        for path in source.iterdir():
            if path.name != "adapter_model.safetensors" and path.is_file():
                shutil.copy2(path, target / path.name)
        save_file(repaired, str(target / "adapter_model.safetensors"))
    except BaseException:
        shutil.rmtree(target)
        raise

    config = json.loads((target / "adapter_config.json").read_text(encoding="utf-8"))
    result = {
        "source": str(source),
        "target": str(target),
        "tensor_count": len(repaired),
        "renamed": changed,
        "rank": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
