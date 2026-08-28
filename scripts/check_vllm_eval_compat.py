#!/usr/bin/env python3
"""Fail fast when the installed vLLM cannot import VERL's rollout stack."""

import importlib

from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.utils.vllm import VLLMHijack


def main() -> None:
    VLLMHijack.hijack()
    if not getattr(LRUCacheWorkerLoRAManager, "_verl_tensor_lora_hijacked", False):
        raise RuntimeError("VERL tensor-LoRA bridge was not installed")
    importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_rollout")
    importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    print("vLLM eval/LoRA/rollout imports: ok")


if __name__ == "__main__":
    main()
