"""Shared validation and vLLM adaptation for structured controller turns."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, TypeVar


GuidedDecodingT = TypeVar("GuidedDecodingT")
_LEADING_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def normalize_structured_content(value: str) -> str:
    """Remove a leading Qwen reasoning envelope from structured JSON.

    Qwen3 may emit even an empty ``<think>...</think>`` block when the chat
    template requests non-thinking mode.  vLLM returns that block in
    ``message.content`` unless a reasoning parser is active, which otherwise
    makes a valid guided JSON object fail controller parsing.
    """
    if not isinstance(value, str):
        raise TypeError("structured response content must be a string")
    return _LEADING_THINK_BLOCK.sub("", value, count=1).strip()


def normalize_structured_outputs(value: Any) -> dict[str, Any]:
    """Validate the controller envelope while keeping it Ray-serializable.

    External vLLM's OpenAI server accepts ``structured_outputs={"json": ...}``,
    while VERL uses offline ``SamplingParams``. The agent loop transports this
    small plain dictionary across Ray; the rollout server adapts it to the
    legacy GuidedDecodingParams or modern StructuredOutputsParams API.
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


def build_vllm_structured_sampling_kwargs(value: Any, sampling_module: Any) -> dict[str, Any]:
    """Adapt the wire schema to the installed vLLM sampling API.

    vLLM 0.12 removed GuidedDecodingParams. Prefer its replacement when
    available, retaining compatibility with the original 0.10 training env.
    """
    normalized = normalize_structured_outputs(value)
    modern = getattr(sampling_module, "StructuredOutputsParams", None)
    if modern is not None:
        return {"structured_outputs": modern(json=normalized["json"])}
    return {"guided_decoding": sampling_module.GuidedDecodingParams(json=normalized["json"])}
