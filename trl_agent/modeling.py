"""Model construction shared with QeRL without importing its reward stack."""

from __future__ import annotations

from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model_and_peft(model_args, training_args):
    """Build the same model, tokenizer, and LoRA configuration as QeRL.

    Importing QeRL's top-level ``qerl`` module also imports its math reward
    implementation.  That reward stack requires ANTLR 4.13, while the
    OmegaConf/Hydra stack used by ContextGraph requires ANTLR 4.9.  Agent
    training uses ``MathEnv``'s strict numeric scorer, so keep model setup
    independent from the unused QeRL reward modules.
    """
    noise_scheduler = "nvfp4" in model_args.model_name.lower() and not model_args.disable_noise

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name,
        attn_implementation="sdpa",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
    )
    peft_config = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.target_modules,
        modules_to_save=(
            ["input_layernorm", "post_attention_layernorm"] if model_args.ln else None
        ),
    )
    return model, tokenizer, peft_config, noise_scheduler
