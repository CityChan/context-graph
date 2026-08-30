"""Utilities for producing and validating PEFT-compatible LoRA adapters."""


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


def assert_no_meta_lora_params(module) -> None:
    """Fail when a loaded PEFT adapter still contains unmaterialized tensors."""
    meta_params = [name for name, param in module.named_parameters() if "lora_" in name and param.is_meta]
    if meta_params:
        preview = ", ".join(meta_params[:5])
        raise RuntimeError(
            f"LoRA adapter load left {len(meta_params)} parameters on the meta device; "
            f"the checkpoint weights were not materialized. First parameters: {preview}"
        )


def assert_no_meta_params(module, *, context: str) -> None:
    """Fail when distributed initialization leaves any model parameter on meta."""
    meta_params = [name for name, param in module.named_parameters() if param.is_meta]
    if meta_params:
        preview = ", ".join(meta_params[:5])
        raise RuntimeError(
            f"{context} left {len(meta_params)} parameters on the meta device. First parameters: {preview}"
        )
