#!/usr/bin/env python3
"""Fail fast when the installed vLLM cannot import VERL's rollout stack."""

import importlib
import json
import os
from pathlib import Path

from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.utils.vllm import VLLMHijack


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no"}


def validate_qwen35_aot_configuration() -> None:
    model_path = Path(os.environ.get("MODEL_PATH", ""))
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return
    model_type = json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
    if model_type != "qwen3_5":
        return
    if _enabled("TORCHDYNAMO_DISABLE"):
        raise RuntimeError(
            "Qwen3.5/3.6 vLLM AOT startup requires TORCHDYNAMO_DISABLE=0"
        )
    if not _enabled("VLLM_USE_AOT_COMPILE"):
        raise RuntimeError(
            "Qwen3.5/3.6 vLLM AOT startup requires VLLM_USE_AOT_COMPILE=1"
        )
    import torch

    if not hasattr(torch._dynamo.config, "enable_aot_compile"):
        raise RuntimeError(
            f"PyTorch {torch.__version__} does not expose enable_aot_compile"
        )


def main() -> None:
    validate_qwen35_aot_configuration()
    VLLMHijack.hijack()
    if not getattr(LRUCacheWorkerLoRAManager, "_verl_tensor_lora_hijacked", False):
        raise RuntimeError("VERL tensor-LoRA bridge was not installed")
    rollout_module = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_rollout")
    rollout_module.validate_worker_wrapper_constructor()
    importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    print("vLLM eval/LoRA/rollout imports, AOT config, and worker constructor: ok")


if __name__ == "__main__":
    main()
