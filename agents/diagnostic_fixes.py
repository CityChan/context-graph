"""Opt-in, evaluation-only interventions; never change task rewards or answers."""

import json
from collections import Counter

ANSWER_CONSISTENCY = (
    "[FINAL ANSWER CONSISTENCY] In the finish tool, put the complete, "
    "evidence-supported name or value in the answer field. Use the same identity "
    "and name in the explanation. Do not replace a supported full name with an "
    "abbreviation, a spouse's surname, or an unverified alias. Do not invent "
    "missing name components. Check these fields for consistency before submitting."
)


def validate_fix(name, is_train):
    if name not in {"none", "answer", "repeat"}:
        raise ValueError(f"Unknown diagnostic fix: {name}")
    if name != "none" and is_train:
        raise ValueError("Diagnostic fixes are evaluation-only")
    return name


class RepeatAdvice:
    """Warn on the third identical main search or no-op focus selection.

    Counts are per episode. This is a prompt intervention, not a tool block,
    semantic similarity detector, or automatic answer/graph mutation.
    """

    def __init__(self):
        self.counts = Counter()
        self.warnings = 0

    def observe(self, function, arguments):
        if function not in {"search", "select_noop"}:
            return ""
        key = (function, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        self.counts[key] += 1
        if self.counts[key] != 3:
            return ""
        self.warnings += 1
        return (
            "[REPEATED ACTION] You have made this exact search or selected the "
            "already-active focus three times. Inspect the evidence already "
            "available. Prefer a different query, a new source or a different "
            "focus that resolves a specific missing fact. If no useful research "
            "step remains, submit your best evidence-supported answer with finish. "
            "Repeating remains allowed when you can justify its value."
        )
