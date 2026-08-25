"""Deterministic normalization for recoverable tool-call formatting errors."""

from __future__ import annotations

import re
from typing import Any


_MISPLACED_PARAMETER_NAME = re.compile(
    r"<parameter>\s*([A-Za-z_][A-Za-z0-9_.-]*)>"
)
_FUNCTION_NAME_ELEMENT = re.compile(
    r"<function>\s*(action)\s*</function>",
    re.IGNORECASE,
)
_PARAMETER_NAME_ATTRIBUTE = re.compile(
    r"<parameter\s+name\s*=\s*(['\"])([A-Za-z_][A-Za-z0-9_.-]*)\1\s*>",
    re.IGNORECASE,
)
_INLINE_ACTION_COMMAND = re.compile(
    r"<function=action>\s*([^<>]+?)\s*</function>",
    re.IGNORECASE | re.DOTALL,
)


def canonicalize_tool_call_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Canonicalize unambiguous XML action-call formatting mistakes.

    DeepSeek-V4 sometimes emits ``<parameter>command>...`` for the documented
    ``<parameter=command>...`` form. It can also express the function name as
    an XML element with a ``name`` parameter attribute, or place a plain-text
    action command directly inside ``<function=action>``. In each supported
    case the action and command are explicit, so the controller can repair the
    syntax without making a semantic decision. Text outside a recognizable
    action call is never rewritten.
    """
    if not isinstance(text, str) or (
        "<function=" not in text and "<function>" not in text
    ):
        return text, []

    repairs: list[dict[str, Any]] = []

    def replace_function_name(match: re.Match[str]) -> str:
        repairs.append({
            "kind": "function_name_element",
            "function": "action",
        })
        return "<function=action>"

    canonical = _FUNCTION_NAME_ELEMENT.sub(replace_function_name, text)

    def replace_parameter_attribute(match: re.Match[str]) -> str:
        name = match.group(2)
        repairs.append({
            "kind": "parameter_name_attribute",
            "parameter": name,
        })
        return f"<parameter={name}>"

    canonical = _PARAMETER_NAME_ATTRIBUTE.sub(
        replace_parameter_attribute,
        canonical,
    )

    def replace_parameter_separator(match: re.Match[str]) -> str:
        name = match.group(1)
        repairs.append({
            "kind": "parameter_name_separator",
            "parameter": name,
        })
        return f"<parameter={name}>"

    canonical = _MISPLACED_PARAMETER_NAME.sub(
        replace_parameter_separator,
        canonical,
    )

    def replace_inline_action(match: re.Match[str]) -> str:
        command = match.group(1).strip()
        repairs.append({
            "kind": "inline_action_command",
            "parameter": "command",
        })
        return (
            "<function=action>"
            f"<parameter=command>{command}</parameter>"
            "</function>"
        )

    canonical = _INLINE_ACTION_COMMAND.sub(replace_inline_action, canonical)
    return canonical, repairs
