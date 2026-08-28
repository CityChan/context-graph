from pathlib import Path


TRAINER = Path("verl/trainer/fsdp_sft_trainer.py")
CONFIG = Path("verl/trainer/config/sft_trainer.yaml")
FSDP_UTILS = Path("verl/utils/fsdp_utils.py")


def test_fsdp_sft_attention_backend_is_configurable():
    trainer = TRAINER.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert "attn_implementation: flash_attention_2" in config
    assert 'self.config.model.get("attn_implementation", "flash_attention_2")' in trainer
    assert "attn_implementation=attn_implementation" in trainer


def test_fsdp2_skips_the_legacy_fsdp1_wrap_policy_builder():
    trainer = TRAINER.read_text(encoding="utf-8")
    strategy = trainer.index("fsdp_strategy = self.config.model.strategy")
    fsdp1_branch = trainer.index('if fsdp_strategy == "fsdp":', strategy)
    wrap_policy = trainer.index("auto_wrap_policy = get_fsdp_wrap_policy(", strategy)
    fsdp2_branch = trainer.index('elif fsdp_strategy == "fsdp2":', strategy)

    assert fsdp1_branch < wrap_policy < fsdp2_branch


def test_fsdp2_accepts_set_valued_no_split_modules():
    fsdp_utils = FSDP_UTILS.read_text(encoding="utf-8")
    apply_fsdp2 = fsdp_utils[fsdp_utils.index("def apply_fsdp2(") : fsdp_utils.index("def get_shard_placement_fn(")]

    assert "list(fsdp_transformer_layer_cls_to_wrap)" in apply_fsdp2
    assert "fsdp_transformer_layer_cls_to_wrap[0]" not in apply_fsdp2
