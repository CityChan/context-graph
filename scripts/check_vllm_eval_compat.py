#!/usr/bin/env python3
"""Fail fast when the installed vLLM cannot import VERL's rollout stack."""

import importlib

from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.utils.vllm import VLLMHijack


def main() -> None:
    VLLMHijack.hijack()
    if not getattr(LRUCacheWorkerLoRAManager, "_verl_tensor_lora_hijacked", False):
        raise RuntimeError("VERL tensor-LoRA bridge was not installed")
    rollout_module = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_rollout")
    rollout_module.validate_worker_wrapper_constructor()
    importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")
    print("vLLM eval/LoRA/rollout imports and worker constructor: ok")


if __name__ == "__main__":
    main()
