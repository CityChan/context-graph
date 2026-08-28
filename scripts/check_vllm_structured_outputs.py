#!/usr/bin/env python3
"""Fail-fast preflight for the vLLM structured-output API used by controllers."""

from agents.graph_controller import merge_decision_schema
from agents.structured_outputs import build_vllm_structured_outputs
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams


def main() -> None:
    structured = build_vllm_structured_outputs(
        {"json": merge_decision_schema([0, 1])},
        StructuredOutputsParams,
    )
    params = SamplingParams(structured_outputs=structured)
    assert params.structured_outputs is not None
    assert params.structured_outputs.json
    print("vLLM structured outputs: ok")


if __name__ == "__main__":
    main()
