"""Shared validation and vLLM adaptation for structured controller turns."""

from __future__ import annotations

import copy
import json
from typing import Any, TypeVar


StructuredOutputsT = TypeVar("StructuredOutputsT")


def normalize_structured_outputs(value: Any) -> dict[str, Any]:
    """Validate the controller envelope while keeping it Ray-serializable.

    The agent loop transports this small plain dictionary across Ray and the
    rollout server converts it to vLLM's native ``StructuredOutputsParams``.
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


def build_vllm_structured_outputs(
    value: Any,
    structured_outputs_cls: type[StructuredOutputsT],
) -> StructuredOutputsT:
    """Convert the wire envelope to vLLM ``StructuredOutputsParams``."""
    normalized = normalize_structured_outputs(value)
    return structured_outputs_cls(json=normalized["json"])
