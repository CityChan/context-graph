"""Deterministic normalization for recoverable tool-call formatting errors."""

from __future__ import annotations

import re
from typing import Any


_MISPLACED_PARAMETER_NAME = re.compile(
    r"<parameter>\s*([A-Za-z_][A-Za-z0-9_.-]*)>"
)


def canonicalize_tool_call_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Canonicalize unambiguous parameter-name separator mistakes.

    DeepSeek-V4 sometimes emits ``<parameter>command>...`` for the documented
    ``<parameter=command>...`` form. The intended key is explicit, so the
    controller can repair syntax without making a semantic decision. Text
    outside a function call is never rewritten.
    """
    if not isinstance(text, str) or "<function=" not in text:
        return text, []

    repairs: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        repairs.append({
            "kind": "parameter_name_separator",
            "parameter": name,
        })
        return f"<parameter={name}>"

    canonical = _MISPLACED_PARAMETER_NAME.sub(replace, text)
    return canonical, repairs
