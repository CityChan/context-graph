from pathlib import Path


SCRIPT = Path("scripts/embed_local_search_corpus.py")


def test_embedding_builder_defaults_to_sdpa_without_flash_attn_dependency():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'default="sdpa"' in text
    assert 'choices=("sdpa", "eager", "flash_attention_2")' in text
    assert "dtype=torch.bfloat16" in text
