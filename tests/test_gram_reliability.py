"""Regression coverage for helper failures and bounded BC-P completion."""
import asyncio
import json
import re

import httpx
import pytest
from jsonschema import Draft202012Validator

from agents.gram_agent import GramConfig, MemoryBackend, is_abstention, run_episode
from agents.gram_bcp import score_bcp
from agents.gram_decoding import action_regex
from agents.gram_memory import GraphMemory, memory_output_schema, validate_memory_output


@pytest.mark.parametrize("operation", ["entities", "relations", "maintenance"])
@pytest.mark.parametrize("overflow", ["items", "string"])
def test_output_bounds_enforced_by_schema_and_parser(operation, overflow):
    rows = ["A"] * 49 if overflow == "items" else ["x" * 257]
    value = rows if operation == "entities" else [[s, "r", "B"] for s in rows]
    if operation == "maintenance":
        value = {"add": value, "remove": []}
    assert not Draft202012Validator(memory_output_schema(operation)).is_valid(value)
    with pytest.raises(ValueError, match="limit"):
        validate_memory_output(operation, value)


def test_graph_removals_strict_by_default_and_atomic_even_when_lenient():
    graph = GraphMemory()
    graph.apply([["A", "r", "B"]], [], "old")
    before = graph.snapshot()
    with pytest.raises(ValueError, match="absent"):
        graph.apply([["C", "r", "D"]], [["Missing", "r", "B"]], "new")
    assert graph.snapshot() == before
    with pytest.raises(ValueError, match="three strings"):
        graph.apply([["malformed"]], [["A", "r", "B"]], "new", skip_absent_removals=True)
    assert graph.snapshot() == before
    skipped = graph.apply([["C", "r", "D"]],
                          [["Missing", "r", "B"], ["A", "r", "B"], ["A", "r", "B"]],
                          "new", skip_absent_removals=True)
    assert skipped == [("Missing", "r", "B")]
    assert graph.snapshot() == [{"triple": ["C", "r", "D"], "sources": ["new"]}]


def test_real_helper_update_audits_skipped_removal_and_requests_compact_json():
    async def run():
        events = []
        backend = MemoryBackend("http://helper", "frozen", audit=events.append)
        await backend.aclose()
        def reply(request):
            body = json.loads(request.content)
            assert body["structured_outputs"]["disable_any_whitespace"] is True
            value = {"add": [["Alice", "lives_in", "Paris"]],
                     "remove": [["Alice", "lives_in", "London"]]}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)},
                                                         "finish_reason": "stop"}]})
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        graph = GraphMemory()
        try:
            edit = await backend.edit("memory_update", "Alice lives in Paris", {"id": "doc"},
                                      "unused", graph, .9)
            assert edit["skipped_absent_removals"] == [("Alice", "lives_in", "London")]
            assert graph.snapshot()[0]["triple"] == ["Alice", "lives_in", "Paris"]
            assert json.loads(events[0]["response"])["remove"] == [["Alice", "lives_in", "London"]]
        finally:
            await backend.aclose()
    asyncio.run(run())


def run_scripted(actions, config, *, spend_full=False):
    states, budgets, events = [], [], []
    async def policy(messages, budget):
        states.append(json.loads(messages[-1]["content"]))
        budgets.append(budget)
        size = budget if spend_full else 2
        return actions[len(states)-1], {"response_ids": [1]*size, "response_mask": [1]*size}
    async def retrieval(*args):
        return "Book was written by Alice"
    class Helper:
        async def edit(self, operation, content, document, question, graph, threshold):
            graph.apply([["Book", "author", "Alice"]], [], document["id"])
            return {}
    result = asyncio.run(run_episode({"task_id": "x", "question": "Who wrote Book?", "documents": []},
                                    policy, Helper(), config, events.append, retrieval=retrieval))
    return result, states, budgets, events


def test_last_step_answer_uses_saved_graph_and_hides_pending_document():
    result, states, _, _ = run_scripted(
        ["<search>Book</search>", "<memory_insert>Book by Alice</memory_insert>",
         "<search>new evidence</search>", "<answer>Alice</answer>"],
        GramConfig(max_steps=4, action_decoding="xml_regex", final_answer_tokens=512))
    assert result["prediction"] == "Alice" and result["steps"] == 4
    assert result["documents_consumed"] == 1
    assert states[-1]["finalizing"] and states[-1]["allowed_actions"] == ["answer"]
    assert states[-1]["document_obs"] is None
    assert states[-1]["memory_obs"]["search_paths"] == [[["Book", "author", "Alice"]]]
    pattern = action_regex(["answer"], allow_think=False)
    assert re.fullmatch(pattern, "<answer>Alice</answer>")
    assert not re.fullmatch(pattern, "<think>long</think><answer>Alice</answer>")
    assert not re.fullmatch(pattern, "<search>Book</search>")


def test_final_answer_reserve_stays_inside_original_token_budget():
    result, states, budgets, _ = run_scripted(
        ["<search>Book</search>", "<answer>Alice</answer>"],
        GramConfig(max_steps=20, max_step_tokens=100, max_episode_tokens=150,
                   action_decoding="xml_regex", final_answer_tokens=50), spend_full=True)
    assert budgets == [100, 50] and states[-1]["finalizing"]
    assert result["policy_tokens"] == 150 and result["steps"] == 2


def test_abstention_does_not_end_episode_before_final_budget():
    result, states, _, events = run_scripted(
        ["<search>Book</search>", "<memory_insert>Book by Alice</memory_insert>",
         "<answer>None</answer>", "<answer>Alice</answer>"],
        GramConfig(max_steps=4, action_decoding="xml_regex", final_answer_tokens=50,
                   reject_abstentions=True, bcp_progress_limit=2))
    assert result["prediction"] == "Alice" and result["abstentions_rejected"] == 1
    assert "Abstention" in events[2]["error"]
    assert "Abstention" in states[3]["memory_obs"]["feedback"]
    result, _, _, _ = run_scripted(["<search>Book</search>", "<answer>unknown</answer>"],
        GramConfig(max_steps=2, action_decoding="xml_regex", final_answer_tokens=50,
                   reject_abstentions=True))
    assert result["prediction"] == "" and result["termination_reason"] == "abstention"
    assert is_abstention("None.") and not is_abstention("And Then There Were None")


@pytest.mark.parametrize("eventual", ["llm_judge", "llm_parse_failure", "offline_strict_only"])
def test_judge_retries_only_transient_failures_with_raw_audit(monkeypatch, capsys, eventual):
    calls, delays = [], []
    async def judge(question, answer, prediction, audit_sink):
        calls.append(1)
        method = "llm_parse_failure" if len(calls) == 1 else eventual
        audit_sink.append({"judge_method": method, "grader_attempts": [{"raw_response": "bad JSON"}]})
        return 0  # Valid negative judgments must never be retried.
    async def sleep(delay):
        delays.append(delay)
    monkeypatch.setattr("agents.gram_bcp.judge", judge)
    monkeypatch.setattr("agents.gram_bcp.asyncio.sleep", sleep)
    if eventual == "llm_judge":
        scored = asyncio.run(score_bcp("q", "a", "p"))
        assert scored["score"] == 0 and len(calls) == 2
        assert [r["outer_attempt"] for r in scored["judge_audit"]] == [1, 2]
    else:
        with pytest.raises(RuntimeError, match="not a policy failure"):
            asyncio.run(score_bcp("q", "a", "p"))
    assert delays == ([15, 60] if eventual == "llm_parse_failure" else [15])
    assert "bad JSON" in capsys.readouterr().out
