import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "verl" / "utils" / "dataset" / "chatml_loss_mask.py"
SPEC = importlib.util.spec_from_file_location("contextgraph_test_chatml_loss_mask", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_chatml_assistant_mask = MODULE.build_chatml_assistant_mask


def test_marks_only_assistant_payloads_and_closing_tokens():
    im_start, im_end = 100, 101
    assistant_header = [7, 8]
    ids = [im_start, 1, 2, 11, im_end, im_start, 7, 8, 21, 22, im_end, im_start, 1, 2, 31, im_end]
    mask, spans = build_chatml_assistant_mask(
        ids, im_start_id=im_start, im_end_id=im_end, assistant_header_ids=assistant_header
    )
    assert spans == 1
    assert [index for index, value in enumerate(mask) if value] == [8, 9, 10]


def test_marks_multiple_assistant_spans():
    ids = [100, 7, 8, 20, 101, 4, 100, 7, 8, 30, 31, 101]
    mask, spans = build_chatml_assistant_mask(ids, im_start_id=100, im_end_id=101, assistant_header_ids=[7, 8])
    assert spans == 2
    assert sum(mask) == 5


def test_rejects_unclosed_assistant_span():
    with pytest.raises(ValueError, match="missing its <\\|im_end\\|> token"):
        build_chatml_assistant_mask([100, 7, 8, 20], im_start_id=100, im_end_id=101, assistant_header_ids=[7, 8])
