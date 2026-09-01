import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_lora_merge_delta.py"
SPEC = importlib.util.spec_from_file_location("check_lora_merge_delta", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_weight_map(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        '{"weight_map":{"model.layers.0.self_attn.q_proj.weight":"model-1.safetensors"}}',
        encoding="utf-8",
    )
    assert MODULE.load_weight_map(tmp_path) == {
        "model.layers.0.self_attn.q_proj.weight": "model-1.safetensors"
    }


def test_load_weight_map_rejects_missing_index(tmp_path):
    try:
        MODULE.load_weight_map(tmp_path)
    except FileNotFoundError as exc:
        assert "missing safetensors index" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
