#!/usr/bin/env python3
"""Merge a VERL-exported LoRA adapter into its Hugging Face base model."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    return parser.parse_args()


def repair_adapter_config(
    adapter_dir: Path, base_model: Path, lora_rank: int, lora_alpha: int
) -> dict:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing LoRA adapter config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("r", -1)) != lora_rank:
        raise ValueError(f"adapter rank {config.get('r')} does not match expected rank {lora_rank}")
    config["lora_alpha"] = lora_alpha
    config["task_type"] = "CAUSAL_LM"
    config["base_model_name_or_path"] = str(base_model)
    config["inference_mode"] = True
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def main() -> None:
    args = parse_args()
    base_model = Path(args.base_model)
    adapter_dir = Path(args.adapter)
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    config = repair_adapter_config(
        adapter_dir,
        base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True)
    model = model.merge_and_unload(safe_merge=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, local_files_only=True)
    tokenizer.save_pretrained(output_dir)
    del model
    gc.collect()

    loaded_config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True, local_files_only=True)
    AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True, local_files_only=True)
    weight_files = sorted(output_dir.glob("*.safetensors"))
    if getattr(loaded_config, "model_type", None) != "qwen3" or not weight_files:
        raise RuntimeError("merged LoRA output failed Hugging Face artifact validation")
    print(
        json.dumps(
            {
                "base_model": str(base_model),
                "adapter": str(adapter_dir),
                "output": str(output_dir),
                "lora_rank": config["r"],
                "lora_alpha": config["lora_alpha"],
                "model_type": loaded_config.model_type,
                "weight_files": len(weight_files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
