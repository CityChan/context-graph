"""State-dependent XML constraints for the explicitly opted-in GRAM protocol."""

OPERATIONS = ("memory_insert", "memory_update", "memory_search", "answer", "search", "open_page")
# No raw XML delimiters, C0 controls, attributes or nested tags. The strict XML
# parser remains authoritative (including Unicode/XML validity and empty bodies).
# Use common Rust/Python regex syntax for the external and offline vLLM backends.
TEXT = r"(?:[^<>&\x00-\x08\x0B\x0C\x0E-\x1F]|&(?:amp|lt|gt|quot|apos);)"
SPACE = r"[ \t\r\n]*"


def allowed_actions(*, has_document, external, external_searches,
                    progress_limit=0, has_memory=True, memory_searches=0):
    """Base executor guards, plus the explicitly enabled BC-P progress guard."""
    actions = ["memory_search"]
    if has_document:
        actions += ["memory_insert", "memory_update"]
    if not external or external_searches:
        actions.append("answer")
    if external and not has_document:
        actions += ["search", "open_page"]
    if external and progress_limit:
        if not has_memory or memory_searches >= progress_limit:
            actions.remove("memory_search")
        if has_document and "answer" in actions:
            actions.remove("answer")
        if not external_searches and "open_page" in actions:
            actions.remove("open_page")
    return actions


def action_regex(actions, *, allow_think=True):
    if not actions or len(actions) != len(set(actions)) or any(a not in OPERATIONS for a in actions):
        raise ValueError("Invalid GRAM allowed_actions")
    branches = "|".join(f"<{a}>{TEXT}+</{a}>" for a in actions)
    thinking = f"(?:<think>{TEXT}*</think>{SPACE})?" if allow_think else ""
    return SPACE + thinking + f"(?:{branches})" + SPACE
