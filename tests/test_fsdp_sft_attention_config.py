from pathlib import Path


TRAINER = Path("verl/trainer/fsdp_sft_trainer.py")
CONFIG = Path("verl/trainer/config/sft_trainer.yaml")


def test_fsdp_sft_attention_backend_is_configurable():
    trainer = TRAINER.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert "attn_implementation: flash_attention_2" in config
    assert 'self.config.model.get("attn_implementation", "flash_attention_2")' in trainer
    assert "attn_implementation=attn_implementation" in trainer
