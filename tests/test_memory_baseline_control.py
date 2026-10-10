import asyncio
import copy
import json
import re

import pytest
from jsonschema import Draft202012Validator

from agents import memory_baseline_agent as runner
from agents.memory_baseline_control import FINAL_SYSTEM, helper_schema, final_constraint, editable_node_ids
from tests.test_graph_memory_baselines import setup, graph, PAIR, PATCH, ANALYSIS, EVOLUTION, NOOP
from tests.test_agentfold import SEARCH, OPEN, FINISH


@pytest.mark.parametrize("phase,value", [("memory_memorize", PATCH), ("memory_recall", NOOP),
    ("memory_analyze", ANALYSIS), ("memory_evolve", EVOLUTION)])
def test_helper_contracts_accept_expected_decisions(phase, value):
    schema = helper_schema(phase, state=graph().state() if phase == "memory_recall" else None)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(value)


def test_schema_rejects_observed_task_kind_and_string_thought():
    check = Draft202012Validator(helper_schema("memory_memorize"))
    patch = copy.deepcopy(PATCH)
    patch["add_nodes"][0]["kind"] = "task"
    assert not check.is_valid(patch)
    patch["add_nodes"][0]["kind"] = "subtask"
    patch["add_nodes"][0]["thought"] = "plain text instead of messages"
    assert not check.is_valid(patch)
    assert check.is_valid({"add_nodes": [], "add_edges": []})


@pytest.mark.parametrize("method,decision", [("memobrain", PATCH), ("amem", ANALYSIS)])
def test_control_modes_are_per_request_and_preserve_accounting(monkeypatch, method, decision):
    env, context, item, client = setup(monkeypatch, method, [SEARCH, json.dumps(decision), FINISH])
    context.config.actor_rollout_ref.rollout.plugin.val_max_turn = 3
    render_calls = []
    original = runner._apply_chat_template
    def render(tokenizer, messages, config, **kwargs):
        render_calls.append((copy.deepcopy(messages), dict(kwargs)))
        return original(tokenizer, messages, config, **kwargs)
    monkeypatch.setattr(runner, "_apply_chat_template", render)
    before = copy.deepcopy(context.config)
    out = asyncio.run(runner.process_item(item, context))
    records = out.extra_fields["model_contexts"]
    assert out.extra_fields["termination_reason"] == "finish"
    assert records[0]["structured_outputs"] is None
    assert records[1]["structured_outputs"] == {"json": helper_schema(records[1]["phase"])}
    assert records[1]["control_enable_thinking"] is False
    assert records[1]["max_tokens"] <= 1024
    assert records[2]["messages"][0]["content"] == FINAL_SYSTEM
    assert records[2]["control_enable_thinking"] is False
    assert re.fullmatch(records[2]["structured_outputs"]["regex"], FINISH)
    for record in records:
        matching = [kw for messages, kw in render_calls if messages == record["messages"]]
        assert matching
        if record["phase"] == "action":
            assert all("enable_thinking" not in kw for kw in matching)
        else:
            assert any(kw.get("enable_thinking") is False for kw in matching)
    stats = out.extra_fields["env_stats"]
    assert stats["main_len"] == sum(stats[k] for k in ("generated_tokens", "observation_tokens", "instruction_tokens"))
    assert stats["generated_tokens"] == sum(len(r["output_ids"]) for r in records)
    assert context.config == before


def test_budget_final_rejects_search_and_retains_retry_feedback(monkeypatch):
    env, context, item, client = setup(monkeypatch, "memobrain", [SEARCH, SEARCH, FINISH])
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.turn_max_new_tokens = 500
    plugin.final_answer_reserve = 1500
    plugin.val_response_length = 2000  # Final phase from the first request.
    out = asyncio.run(runner.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "finish"
    assert env.actions == [FINISH]  # A server ignoring the constraint cannot execute searches.
    records = out.extra_fields["model_contexts"]
    assert all(r["phase"] == "final" for r in records)
    assert "Final answer violated" in str(records[1]["messages"])
    assert out.extra_fields["env_stats"]["invalid_tool"] == 2


def test_semantically_invalid_memory_remains_a_failure(monkeypatch):
    invalid = dict(add_nodes=[], add_edges=[dict(src=1, dst=999, rationale="unknown")])
    Draft202012Validator(helper_schema("memory_memorize")).validate(invalid)
    env, context, item, _ = setup(monkeypatch, "memobrain", [SEARCH, json.dumps(invalid), FINISH])
    out = asyncio.run(runner.process_item(item, context))
    assert out.extra_fields["env_stats"]["invalid_memory"] == 1
    assert not out.extra_fields["memory_audit"][0]["applied"]
    assert out.extra_fields["memory_state"]["edges"] == []
    assert env.actions == [SEARCH, FINISH]


def test_final_constraint_keeps_budget_bound_and_forbids_tools():
    assert final_constraint(96) is None
    regex = final_constraint(110)["regex"]
    assert re.fullmatch(regex, FINISH)
    assert not re.fullmatch(regex, SEARCH)
    assert not re.fullmatch(regex, FINISH.replace("gold", "x" * 15))


def test_ignored_final_constraints_still_stop_after_three_rejections(monkeypatch):
    env, context, item, _ = setup(monkeypatch, "memobrain", [SEARCH] * 3)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.turn_max_new_tokens = 5000
    plugin.val_response_length = 5000
    out = asyncio.run(runner.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "invalid_tool_limit"
    assert out.extra_fields["env_stats"]["model_requests"] == 3
    assert not env.actions


def test_no_unconstrained_final_when_budget_cannot_fit_contract(monkeypatch):
    env, context, item, client = setup(monkeypatch, "memobrain", [])
    context.config.actor_rollout_ref.rollout.plugin.val_response_length = 90
    out = asyncio.run(runner.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "token_limit"
    assert not client.calls and not env.actions


def test_recall_targets_refresh_after_folding_and_protection_changes():
    store = graph()
    before = helper_schema("memory_recall", state=store.state())
    assert editable_node_ids(store.state()) == [3, 4, 5]
    check = Draft202012Validator(before)
    fold = dict(flush_ops=[], fold_ops=[dict(ids=[3, 4], rationale="done", notes=PAIR)])
    check.validate(fold)
    for nid in (1, 2, 6, 7, 999):
        assert not check.is_valid(dict(flush_ops=[dict(id=nid, rationale="done")], fold_ops=[]))
        assert not check.is_valid(dict(flush_ops=[], fold_ops=[dict(ids=[nid], rationale="done", notes=PAIR)]))
    store.maintain(fold)
    store.append(PAIR)
    store.patch(PATCH)
    state = json.loads(json.dumps(store.state()))  # Serialized node keys are strings.
    assert editable_node_ids(state) == [5, 6, 8]
    after = helper_schema("memory_recall", state=state)
    assert after["properties"]["flush_ops"]["items"]["properties"]["id"]["enum"] == [5, 6, 8]
    assert before["properties"]["flush_ops"]["items"]["properties"]["id"]["enum"] == [3, 4, 5]
    assert not Draft202012Validator(after).is_valid(fold)


def test_recall_without_editable_nodes_only_allows_noop():
    schema = helper_schema("memory_recall", state=graph(2).state())
    Draft202012Validator.check_schema(schema)
    check = Draft202012Validator(schema)
    check.validate(NOOP)
    assert not check.is_valid(dict(flush_ops=[dict(id=1, rationale="done")], fold_ops=[]))
    assert not check.is_valid(dict(flush_ops=[], fold_ops=[dict(ids=[1], rationale="done", notes=PAIR)]))
    with pytest.raises(ValueError, match="current graph state"):
        helper_schema("memory_recall")


def test_recall_snapshot_is_sent_and_illegal_server_response_still_rejected(monkeypatch):
    invalid = dict(flush_ops=[dict(id=2, rationale="done")], fold_ops=[])
    env, context, item, _ = setup(monkeypatch, "memobrain",
        [SEARCH, json.dumps(PATCH), json.dumps(invalid), FINISH])
    context.config.actor_rollout_ref.rollout.plugin.memobrain_recall_interval = 1
    out = asyncio.run(runner.process_item(item, context))
    records = out.extra_fields["model_contexts"]
    recall = next(r for r in records if r["phase"] == "memory_recall")
    state = json.loads(recall["messages"][1]["content"])
    assert "Editable node IDs: []" in recall["messages"][0]["content"]
    assert recall["structured_outputs"] == {"json": helper_schema("memory_recall", state=state)}
    assert out.extra_fields["env_stats"]["invalid_memory"] == 1
    assert out.extra_fields["env_stats"]["memory_requests"] == 2  # No hidden retries.
    assert json.loads(json.dumps(out.extra_fields["memory_state"])) == state
    assert env.actions == [SEARCH, FINISH]


def test_recall_overlap_remains_transactionally_rejected():
    store = graph()
    before = copy.deepcopy(store.state())
    overlap = dict(flush_ops=[dict(id=3, rationale="old")],
                   fold_ops=[dict(ids=[3, 4], rationale="done", notes=PAIR)])
    # Per-ID legality does not establish cross-operation consistency.
    Draft202012Validator(helper_schema("memory_recall", state=before)).validate(overlap)
    with pytest.raises(ValueError, match="overlapping"):
        store.maintain(overlap)
    assert store.state() == before
