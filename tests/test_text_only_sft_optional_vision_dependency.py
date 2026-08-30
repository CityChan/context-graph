import importlib.util
from pathlib import Path
import sys
import types

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VISION_UTILS_PATH = ROOT / "verl" / "utils" / "dataset" / "vision_utils.py"
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    sys.modules["torch"] = torch_stub
SPEC = importlib.util.spec_from_file_location("contextgraph_test_vision_utils", VISION_UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
vision_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vision_utils)


def test_pil_image_path_does_not_need_qwen_vl_utils():
    image = Image.new("RGB", (2, 2))
    converted = vision_utils.process_image(image)
    assert converted.mode == "RGB"


def test_sft_smoke_preflight_does_not_require_qwen_vl_utils():
    script = (ROOT / "scripts" / "smoke_train_contextgraph_sft_qwen36_27b_4node_idev.sh").read_text()
    preflight_line = next(line for line in script.splitlines() if "node preflight:" in line)
    assert "qwen_vl_utils" not in preflight_line
