"""Shared validation and vLLM adaptation for structured controller turns."""

from __future__ import annotations

import copy
import json
from typing import Any, TypeVar


GuidedDecodingT = TypeVar("GuidedDecodingT")


def normalize_structured_outputs(value: Any) -> dict[str, Any]:
    """Validate the controller envelope while keeping it Ray-serializable.

    External vLLM's OpenAI server accepts ``structured_outputs={"json": ...}``,
    while the vendored VERL stack uses vLLM 0.10.1 offline ``SamplingParams``.
    The agent loop therefore transports this small plain dictionary across Ray
    and converts it to ``GuidedDecodingParams`` inside the rollout server.
    """
    if not isinstance(value, dict):
        raise TypeError("structured_outputs must be a dictionary")
    if set(value) != {"json"}:
        raise ValueError("structured_outputs must contain exactly one 'json' schema")
    schema = value["json"]
    if not isinstance(schema, (dict, str)):
        raise TypeError("structured_outputs['json'] must be a dictionary or JSON string")
    if isinstance(schema, dict):
        # Fail before the request crosses Ray if the schema is not JSON-safe.
        json.dumps(schema)
    return {"json": copy.deepcopy(schema)}


def build_vllm_guided_decoding(
    value: Any,
    guided_decoding_cls: type[GuidedDecodingT],
) -> GuidedDecodingT:
    """Convert the wire envelope to vLLM 0.10.1 GuidedDecodingParams."""
    normalized = normalize_structured_outputs(value)
    return guided_decoding_cls(json=normalized["json"])
