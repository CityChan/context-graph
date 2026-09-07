import json

from envs.search_skill_bank import (
    detect_search_task_type,
    format_search_skills,
    load_skill_bank,
)


def test_loader_accepts_published_skillrl_search_schema(tmp_path):
    path = tmp_path / "skills.json"
    path.write_text(
        json.dumps(
            {
                "general_skills": [{"title": "General", "principle": "Search precisely."}],
                "query_type_skills": {
                    "direct_retrieval": [
                        {
                            "title": "Direct",
                            "principle": "Find the fact.",
                            "when_to_apply": "For direct questions.",
                        }
                    ]
                },
                "common_mistakes": [
                    {"description": "Guessing", "how_to_avoid": "Use evidence."}
                ],
            }
        ),
        encoding="utf-8",
    )

    bank = load_skill_bank(str(path))
    assert bank["task_specific_skills"] == bank["query_type_skills"]
    prompt = format_search_skills("Name the capital of France.", str(path), top_k=1)
    assert "Search precisely" in prompt
    assert "Find the fact" in prompt
    assert "Use evidence" in prompt


def test_search_task_type_detection_covers_multihop_and_comparison():
    assert detect_search_task_type("Who is older, Ada or Charles?") == "comparison"
    assert (
        detect_search_task_type("What nationality was the director of the film?")
        == "multi_hop_reasoning"
    )
    assert detect_search_task_type("When was Ada Lovelace born?") == "entity_attribute_lookup"
