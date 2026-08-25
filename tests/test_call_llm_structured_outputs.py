import pytest

from agents.structured_outputs import (
    build_vllm_guided_decoding,
    normalize_structured_outputs,
)


class _GuidedDecodingParams:
    def __init__(self, *, json):
        self.json = json


def test_vllm_adapter_builds_guided_decoding_only_at_server_boundary():
    schema = {"type": "object", "required": ["choice"]}
    guided = build_vllm_guided_decoding(
        {"json": schema},
        _GuidedDecodingParams,
    )
    assert isinstance(guided, _GuidedDecodingParams)
    assert guided.json == schema
    assert guided.json is not schema


@pytest.mark.parametrize(
    "value",
    [None, {"regex": "x"}, {"json": {}, "regex": "x"}, {"json": 3}],
)
def test_structured_outputs_reject_unsupported_envelopes(value):
    with pytest.raises((TypeError, ValueError)):
        normalize_structured_outputs(value)
