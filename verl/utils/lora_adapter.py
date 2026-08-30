"""Utilities for producing PEFT-compatible LoRA adapter state dicts."""


def normalize_lora_adapter_key(name: str, model_type: str | None = None) -> str:
    """Convert checkpoint parameter names to PEFT adapter serialization names."""
    name = name.removeprefix("_fsdp_wrapped_module.")
    name = name.replace(".default.weight", ".weight")

    # Qwen3.5 checkpoints created before the Transformers v5 module-layout
    # conversion use ``model.layers``. The current conditional-generation
    # model nests the text backbone under ``model.language_model.layers``.
    if model_type == "qwen3_5":
        legacy_prefix = "base_model.model.model.layers."
        current_prefix = "base_model.model.model.language_model.layers."
        if name.startswith(legacy_prefix):
            name = current_prefix + name.removeprefix(legacy_prefix)

    return name
