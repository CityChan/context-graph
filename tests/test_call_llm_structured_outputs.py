import pytest

from agents.structured_outputs import (
    build_vllm_structured_outputs,
    normalize_structured_outputs,
)


class _StructuredOutputsParams:
    def __init__(self, *, json):
        self.json = json


def test_vllm_adapter_builds_structured_outputs_only_at_server_boundary():
    schema = {"type": "object", "required": ["choice"]}
    structured = build_vllm_structured_outputs(
        {"json": schema},
        _StructuredOutputsParams,
    )
    assert isinstance(structured, _StructuredOutputsParams)
    assert structured.json == schema
    assert structured.json is not schema


@pytest.mark.parametrize(
    "value",
    [None, {"regex": "x"}, {"json": {}, "regex": "x"}, {"json": 3}],
)
def test_structured_outputs_reject_unsupported_envelopes(value):
    with pytest.raises((TypeError, ValueError)):
        normalize_structured_outputs(value)
