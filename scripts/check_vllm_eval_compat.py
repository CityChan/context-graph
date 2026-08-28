#!/usr/bin/env python3
"""Fail fast when the installed vLLM cannot import VERL's LoRA bridge."""

from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

from verl.utils.vllm import VLLMHijack


def main() -> None:
    VLLMHijack.hijack()
    if not getattr(LRUCacheWorkerLoRAManager, "_verl_tensor_lora_hijacked", False):
        raise RuntimeError("VERL tensor-LoRA bridge was not installed")
    print("vLLM eval/LoRA bridge: ok")


if __name__ == "__main__":
    main()
