"""Static Search SkillBank retrieval compatible with SkillRL's JSON schema."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=8)
def load_skill_bank(path: str) -> dict[str, Any]:
    skill_path = Path(path).expanduser().resolve()
    if not skill_path.is_file():
        raise FileNotFoundError(f"Search SkillBank file not found: {skill_path}")
    with skill_path.open("r", encoding="utf-8") as handle:
        bank = json.load(handle)
    if not isinstance(bank.get("general_skills"), list):
        raise ValueError(f"Search SkillBank has no general_skills list: {skill_path}")
    # The published Search bank uses query_type_skills while newer SkillRL
    # memory code calls the same mapping task_specific_skills.
    task_skills = bank.get("task_specific_skills", bank.get("query_type_skills"))
    if not isinstance(task_skills, dict):
        raise ValueError(f"Search SkillBank has no query/task-specific skill mapping: {skill_path}")
    bank["task_specific_skills"] = task_skills
    return bank


def detect_search_task_type(question: str) -> str:
    """Choose one of the four categories in the published Search SkillBank."""
    query = question.casefold()
    comparison_terms = (
        "compare",
        "difference between",
        "which is older",
        "which was first",
        "which came first",
        "which is larger",
        "which is longer",
        "which is higher",
        "which is lower",
        "who is older",
        "both",
    )
    if any(term in query for term in comparison_terms):
        return "comparison"

    multi_hop_terms = (
        "whose ",
        "the author of",
        "the director of",
        "the founder of",
        "the creator of",
        "the composer of",
        "the country where",
        "the city where",
        "was born",
        "is married to",
        "known for",
    )
    if any(term in query for term in multi_hop_terms):
        return "multi_hop_reasoning"

    attribute_starts = (
        "when ",
        "where ",
        "who ",
        "what year ",
        "what date ",
        "what nationality ",
        "how old ",
        "how many ",
    )
    if query.startswith(attribute_starts):
        return "entity_attribute_lookup"
    return "direct_retrieval"


def format_search_skills(question: str, path: str, top_k: int = 6) -> str:
    """Retrieve and format static skills for post-action prompt injection."""
    bank = load_skill_bank(path)
    task_type = detect_search_task_type(question)
    general_skills = bank["general_skills"][: max(0, int(top_k))]
    task_skills = bank["task_specific_skills"].get(task_type, [])
    mistakes = bank.get("common_mistakes", [])[:5]
    sections = ["<search_skills>", "### General Principles"]
    for skill in general_skills:
        sections.append(f"- **{skill.get('title', '')}**: {skill.get('principle', '')}")

    sections.append(f"\n### {task_type.replace('_', ' ').title()} Skills")
    for skill in task_skills:
        sections.append(f"- **{skill.get('title', '')}**: {skill.get('principle', '')}")
        when = skill.get("when_to_apply", "")
        if when:
            sections.append(f"  _Apply when: {when}_")

    if mistakes:
        sections.append("\n### Mistakes to Avoid")
        for mistake in mistakes:
            sections.append(f"- **Don't**: {mistake.get('description', '')}")
            fix = mistake.get("how_to_avoid", "")
            if fix:
                sections.append(f"  **Instead**: {fix}")
    sections.append("</search_skills>")
    return "\n".join(sections)
