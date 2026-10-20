from pathlib import Path


TRAINER = Path("verl/trainer/fsdp_sft_trainer.py")
CONFIG = Path("verl/trainer/config/sft_trainer.yaml")


def test_fsdp_sft_attention_backend_is_configurable():
    trainer = TRAINER.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert "attn_implementation: flash_attention_2" in config
    assert 'self.config.model.get("attn_implementation", "flash_attention_2")' in trainer
    assert "attn_implementation=attn_implementation" in trainer


def test_sft_loader_supports_text_and_conditional_generation_configs():
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, Qwen2Config, Qwen2_5_VLConfig
    from verl.utils.model import get_hf_auto_model_class
    text = Qwen2Config(architectures=["Qwen2ForCausalLM"])
    conditional = Qwen2_5_VLConfig(architectures=["Qwen2_5_VLForConditionalGeneration"])
    assert get_hf_auto_model_class(text) is AutoModelForCausalLM
    assert get_hf_auto_model_class(conditional) is AutoModelForImageTextToText
    assert "model_class = get_hf_auto_model_class(config)" in TRAINER.read_text(encoding="utf-8")
