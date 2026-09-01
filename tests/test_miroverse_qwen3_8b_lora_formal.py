import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MERGE_SCRIPT = ROOT / "scripts" / "merge_contextgraph_lora_hf.py"
WRAPPER = ROOT / "scripts" / "train_miroverse_controller_sft_qwen3_8b_lora32_4k_4node_12h.sh"
SPEC = importlib.util.spec_from_file_location("contextgraph_test_lora_merge", MERGE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_repairs_verl_lora_adapter_config(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config_path = adapter / "adapter_config.json"
    config_path.write_text(json.dumps({"r": 32, "lora_alpha": 0, "task_type": None}))
    config = MODULE.repair_adapter_config(adapter, tmp_path / "base", 32, 64)
    assert config["lora_alpha"] == 64
    assert config["task_type"] == "CAUSAL_LM"
    assert config["inference_mode"] is True


def test_formal_lora_wrapper_has_verified_training_and_merge_contract():
    text = WRAPPER.read_text()
    assert "#SBATCH -N 4" in text
    assert "#SBATCH -t 12:00:00" in text
    assert "export MAX_LENGTH=4096" in text
    assert "export LORA_RANK=32" in text
    assert "export LORA_ALPHA=64" in text
    assert "TRAIN_LR=${TRAIN_LR:-1e-4}" in text
    assert "merge_contextgraph_lora_hf.py" in text
    assert "merged_hf_model_path.txt" in text
