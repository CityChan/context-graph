import json

from agents.context_graph import ContextGraph, NodeType
from agents.graph_controller import GraphActionController
from agents.structured_memory import (
    GapStepScheduler,
    StructuredFactMemory,
    coerce_bool,
    fact_extraction_schema,
    gap_analysis_schema,
    plan_initialization_schema,
    parse_json_object,
)


def test_bool_settings_handle_hydra_and_environment_representations():
    for value in (True, 1, "1", "true", "TRUE", "yes", "on"):
        assert coerce_bool(value) is True
    for value in (False, 0, "0", "false", "FALSE", "no", "off", ""):
        assert coerce_bool(value) is False
    assert coerce_bool(None, default=True) is True


def _fact(subject, predicate, obj, importance="important", confidence=0.8):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "importance": importance,
        "confidence": confidence,
        "quote": f"{subject} {predicate} {obj}",
        "duplicate_of": "",
    }


def test_schemas_require_bounded_structured_outputs():
    fact_schema = fact_extraction_schema(4)
    assert fact_schema["properties"]["facts"]["maxItems"] == 4
    assert fact_schema["properties"]["facts"]["items"]["additionalProperties"] is False
    assert fact_schema["properties"]["links"]["maxItems"] == 8
    assert "duplicate_of" in fact_schema["properties"]["facts"]["items"]["required"]
    assert set(gap_analysis_schema()["required"]) == {
        "covered_aspects", "missing_information", "suggested_searches",
        "can_answer", "confidence", "reasoning",
    }
    assert plan_initialization_schema(3)["properties"]["goals"]["maxItems"] == 3


def test_parse_json_object_handles_thinking_and_fences():
    payload = {"facts": [], "links": []}
    text = f"<think>ignored</think>\n```json\n{json.dumps(payload)}\n```"
    assert parse_json_object(text) == payload
    assert parse_json_object("no json") is None


def test_parse_json_object_repairs_python_dict_and_skips_malformed_prefix():
    text = "Here's a bad object {broken}\n{'facts': [], 'links': []}"
    assert parse_json_object(text) == {"facts": [], "links": []}


def test_gap_step_scheduler_is_deterministic_bounded_and_step_based():
    first = GapStepScheduler(interval=8, jitter=1, seed_text="same task")
    second = GapStepScheduler(interval=8, jitter=1, seed_text="same task")
    assert first.next_turn == second.next_turn
    assert 7 <= first.next_turn <= 9
    due_turn = first.next_turn
    assert first.due(due_turn)
    first.mark_attempt(due_turn)
    second.mark_attempt(due_turn)
    assert first.next_turn == second.next_turn
    assert due_turn + 7 <= first.next_turn <= due_turn + 9


def test_fact_memory_deduplicates_and_preserves_provenance():
    memory = StructuredFactMemory("Who won?", max_facts=8)
    first = memory.add_extraction(
        {"facts": [_fact("Alice", "won", "the prize")], "links": []},
        evidence_node_id="n2",
        source_metadata={"url": "https://example.test/a"},
    )
    second = memory.add_extraction(
        {"facts": [_fact(" alice ", "WON", "the prize", confidence=0.9)], "links": []},
        evidence_node_id="n3",
        source_metadata={"query": "prize winner"},
    )
    assert first["added"] == 1
    assert second["deduplicated"] == 1
    assert second["changed"] is True
    assert second["revision"] == 2
    assert len(memory.facts) == 1
    fact = memory.facts["f1"]
    assert fact.evidence_node_ids == ["n2", "n3"]
    assert fact.confidence == 0.9
    assert fact.sources == [
        "url=https://example.test/a", "query=prize winner",
    ]


def test_semantic_duplicate_id_merges_paraphrase_and_observation_provenance():
    memory = StructuredFactMemory("Who won?", max_facts=8)
    memory.initialize_plan({"plan": "Find the winner", "goals": ["Verify name"]})
    memory.record_observation(
        evidence_node_id="n2", tool_name="open_page", observation="Alice won."
    )
    memory.add_extraction(
        {"facts": [_fact("Alice", "won", "the prize")], "links": []},
        evidence_node_id="n2",
    )
    memory.record_observation(
        evidence_node_id="n3", tool_name="search", observation="Prize goes to Alice."
    )
    paraphrase = _fact("The prize", "was awarded to", "Alice")
    paraphrase["duplicate_of"] = "f1"
    result = memory.add_extraction(
        {"facts": [paraphrase], "links": []}, evidence_node_id="n3"
    )

    assert result["deduplicated"] == 1
    assert len(memory.facts) == 1
    assert memory.facts["f1"].evidence_node_ids == ["n2", "n3"]
    exported = memory.to_dict()
    assert {node["node_type"] for node in exported["nodes"]} == {
        "plan", "goal", "action", "observation", "fact",
    }
    assert {edge["relation"] for edge in exported["edges"]} >= {
        "contains", "produces",
    }


def test_links_are_runtime_validated_and_context_is_bounded():
    memory = StructuredFactMemory("Compare release dates", max_facts=8)
    memory.add_extraction(
        {"facts": [_fact("Work A", "released in", "1999")], "links": []},
        evidence_node_id="n2",
    )
    result = memory.add_extraction(
        {
            "facts": [_fact("Work B", "released in", "2001", "critical")],
            "links": [
                {
                    "new_fact_index": 0,
                    "existing_fact_id": "f1",
                    "relation": "related",
                    "reason": "Both are release dates",
                },
                {
                    "new_fact_index": 0,
                    "existing_fact_id": "not-a-fact",
                    "relation": "supports",
                    "reason": "invalid target",
                },
            ],
        },
        evidence_node_id="n3",
    )
    assert result["links_added"] == 1
    assert len(memory.links) == 1
    rendered = memory.render_context("Work B", max_facts=1, max_tokens=40)
    assert "[f2]" in rendered
    assert "evidence=n3" in rendered
    assert len(rendered.split()) <= 40
    full_render = memory.render_context("release dates", max_facts=4, max_tokens=200)
    assert "[Fact relationships]" in full_render
    assert "[f2] related [f1]" in full_render


def test_gap_state_is_injected_only_when_answer_is_incomplete():
    memory = StructuredFactMemory("Find the inventor")
    memory.update_gaps({
        "covered_aspects": [],
        "missing_information": ["The inventor's full name"],
        "suggested_searches": ["device inventor primary source"],
        "can_answer": False,
        "confidence": "low",
        "reasoning": "Name is missing.",
    })
    context = memory.render_context("inventor")
    assert "[Information gaps]" in context
    assert "[Next-action guidance:" in context
    memory.update_gaps({
        "covered_aspects": ["Identity"],
        "missing_information": [],
        "suggested_searches": ["unneeded search"],
        "can_answer": True,
        "confidence": "high",
        "reasoning": "Sufficient.",
    })
    assert "[Next-action guidance:" not in memory.render_context("inventor")
    assert "[Readiness]" in memory.render_context("inventor")
    assert memory.ready_to_answer is True


def test_gap_fallback_is_conservative_and_goal_directed():
    memory = StructuredFactMemory("Identify the inventor")
    memory.initialize_plan({
        "plan": "Find and verify the inventor",
        "goals": ["Find the inventor's full name", "Verify with a primary source"],
    })
    payload = memory.fallback_gap_analysis()
    assert payload["can_answer"] is False
    assert payload["confidence"] == "low"
    assert len(payload["missing_information"]) == 2
    assert payload["suggested_searches"][0] == "Find the inventor's full name"
    memory.update_gaps(payload)
    assert memory.ready_to_answer is False


def test_sidecar_facts_do_not_expand_graph_action_candidates():
    graph = ContextGraph()
    graph.add_node("question", NodeType.QUERY)
    graph.add_node("raw observation", NodeType.OBSERVATION)
    controller = GraphActionController()
    before = controller.snapshot(graph)

    memory = StructuredFactMemory("question")
    memory.add_extraction(
        {"facts": [_fact("Entity", "has", "value")], "links": []},
        evidence_node_id="n2",
    )

    after = controller.snapshot(graph)
    assert before == after
    assert [candidate.node_id for candidate in after.candidates] == ["n2"]


def test_eval_launchers_wire_structured_memory_without_disabling_controller():
    paths = [
        "scripts/eval_bc_baseline_8b_4node_zeroshot.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
    ]
    for path in paths:
        text = open(path, encoding="utf-8").read()
        assert "plugin.structured_graph_controller" in text
        assert "plugin.structured_memory_enabled" in text
        assert "plugin.structured_memory_required" in text
        assert "plugin.structured_memory_context_budget" in text
        assert "plugin.structured_memory_max_facts" in text
        assert "plugin.structured_memory_plan_max_tokens" in text
        assert "plugin.structured_memory_relation_candidates" in text
        assert "plugin.structured_memory_controller_retries" in text
        assert "plugin.structured_memory_gap_jitter" in text
        assert "plugin.structured_memory_stop_on_ready" in text
        assert "plugin.structured_memory_step_limit" in text

    wrapper = open(
        "scripts/eval_bc_gaia_qwen3_8b_base_idev.sh", encoding="utf-8"
    ).read()
    assert "STRUCTURED_MEMORY_ENABLED=${STRUCTURED_MEMORY_ENABLED:-0}" in wrapper
    assert "STRUCTURED_MEMORY_REQUIRED=${STRUCTURED_MEMORY_REQUIRED:-$STRUCTURED_MEMORY_ENABLED}" in wrapper
    assert wrapper.count("export STRUCTURED_MEMORY_ENABLED") == 2
    assert wrapper.count("export STRUCTURED_MEMORY_CONTROLLER_RETRIES") == 2
    assert wrapper.count("export STRUCTURED_MEMORY_STOP_ON_READY") == 2

    submitter = open(
        "scripts/submit_eval_gaia_structmem_qwen3_8b_4node.sh", encoding="utf-8"
    ).read()
    assert "#SBATCH -N 4" in submitter
    assert "export RUN_BC=0" in submitter
    assert "export RUN_GAIA=1" in submitter
    assert "export STRUCTURED_MEMORY_ENABLED=1" in submitter
    assert "export STRUCTURED_MEMORY_STEP_LIMIT=40" in submitter
