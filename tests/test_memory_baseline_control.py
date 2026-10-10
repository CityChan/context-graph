import asyncio
import copy
import json
import re
from itertools import combinations, product

import pytest
from jsonschema import Draft202012Validator

from agents import memory_baseline_agent as runner, graph_memory_baselines as memory
from agents.memory_baseline_control import FINAL_SYSTEM, helper_schema, final_constraint, editable_node_ids, recall_constraint
from tests.test_graph_memory_baselines import setup, graph, PAIR, PATCH, ANALYSIS, EVOLUTION, NOOP, fake_embed
from tests.test_agentfold import SEARCH, OPEN, FINISH


@pytest.mark.parametrize("phase,value", [("memory_memorize", PATCH), ("memory_recall", NOOP),
    ("memory_analyze", ANALYSIS), ("memory_evolve", EVOLUTION)])
def test_helper_contracts_accept_expected_decisions(phase, value):
    state = graph().state() if phase == "memory_recall" else (
        dict(neighbors=[dict(id="n1")]) if phase == "memory_evolve" else None)
    schema = helper_schema(phase, state=state)
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
    assert recall["structured_outputs"] == recall_constraint(state)
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


def test_recall_regex_exhaustively_accepts_only_sorted_nonempty_editable_subsets():
    store = graph()
    pattern = recall_constraint(store.state())["regex"]
    for length in range(5):
        for ids in product([1, 3, 4, 5, 7, 33], repeat=length):
            decision = dict(flush_ops=[], fold_ops=[dict(ids=list(ids), rationale="done", notes=PAIR)])
            valid = bool(ids) and list(ids) == sorted(set(ids)) and set(ids) <= {3, 4, 5}
            assert bool(re.fullmatch(pattern, json.dumps(decision))) == valid, ids
    # The exact observed regression, regardless of whitespace.
    bad = dict(flush_ops=[], fold_ops=[dict(ids=[3, 4, 4], rationale="done", notes=PAIR)])
    assert not re.fullmatch(pattern, json.dumps(bad, separators=(",", ":")))


def test_recall_regex_supports_all_subsets_multidigit_ids_and_json_escapes():
    state = dict(nodes={i: dict(active=True) for i in (3, 4, 14, 40)}, protected=[])
    pattern = recall_constraint(state)["regex"]
    notes = [dict(role="assistant", content='Evidence "quoted" \\path\n\t中文 😀')]
    for length in range(1, 5):
        for ids in combinations([3, 4, 14, 40], length):
            decision = dict(flush_ops=[], fold_ops=[dict(ids=list(ids), rationale="done", notes=notes)])
            for escaped in (True, False):
                assert re.fullmatch(pattern, json.dumps(decision, ensure_ascii=escaped, indent=2))
    assert re.fullmatch(pattern, json.dumps(NOOP))
    empty = recall_constraint(graph(2).state())["regex"]
    assert re.fullmatch(empty, json.dumps(NOOP))
    assert not re.fullmatch(empty, json.dumps(decision))


def test_recall_regex_keeps_multiple_operations_and_backend_overlap_check():
    store = graph()
    decision = dict(flush_ops=[dict(id=3, rationale="old")],
                    fold_ops=[dict(ids=[4], rationale="done", notes=PAIR),
                              dict(ids=[5], rationale="done", notes=PAIR)])
    pattern = recall_constraint(store.state())["regex"]
    assert re.fullmatch(pattern, json.dumps(decision))
    overlap = copy.deepcopy(decision)
    overlap["fold_ops"][0]["ids"] = [3, 4]
    assert re.fullmatch(pattern, json.dumps(overlap))
    before = copy.deepcopy(store.state())
    with pytest.raises(ValueError, match="overlapping"):
        store.maintain(overlap)
    assert store.state() == before
    store.maintain(decision)
    assert all(not store.nodes[i]["active"] for i in (3, 4, 5))


def test_recall_regex_size_is_quadratic_not_subset_enumeration():
    state = dict(nodes={i: dict(active=True) for i in range(1, 101)}, protected=[1])
    pattern = recall_constraint(state)["regex"]
    assert len(pattern) < 400_000
    decision = dict(flush_ops=[], fold_ops=[dict(ids=list(range(2, 101)), rationale="done", notes=PAIR)])
    assert re.fullmatch(pattern, json.dumps(decision))


@pytest.mark.parametrize("count", [1, 2, 5])
def test_evolution_schema_binds_both_arrays_to_all_neighbors(count):
    state = dict(neighbors=[dict(id=f"n{10 + i}") for i in range(count)])
    schema = helper_schema("memory_evolve", state=state)
    Draft202012Validator.check_schema(schema)
    check = Draft202012Validator(schema)
    decision = copy.deepcopy(EVOLUTION)
    decision["suggested_connections"] = ["n10"]  # A subset of links, not array slots.
    decision["new_context_neighborhood"] = [f"context {i}" for i in range(count)]
    decision["new_tags_neighborhood"] = [[f"tag {i}"] for i in range(count)]
    check.validate(decision)
    reverse = copy.deepcopy(decision)
    reverse["actions"].reverse()
    check.validate(reverse)
    for field in ("new_context_neighborhood", "new_tags_neighborhood"):
        for wrong_count in (count - 1, count + 1):
            bad = copy.deepcopy(decision)
            bad[field] = [decision[field][0]] * wrong_count
            assert not check.is_valid(bad)
    bad = copy.deepcopy(decision)
    bad["suggested_connections"] = ["n1"]
    assert not check.is_valid(bad)


def test_evolution_schema_preserves_noop_and_independent_actions():
    state = dict(neighbors=[dict(id="n1")])
    check = Draft202012Validator(helper_schema("memory_evolve", state=state))
    noop = dict(should_evolve=False, actions=[], suggested_connections=[], tags_to_update=[],
                new_context_neighborhood=[], new_tags_neighborhood=[])
    check.validate(noop)
    strengthen = {**noop, "should_evolve": True, "actions": ["strengthen"],
                  "suggested_connections": ["n1"], "tags_to_update": ["linked"]}
    update = {**noop, "should_evolve": True, "actions": ["update_neighbor"],
              "new_context_neighborhood": ["context"], "new_tags_neighborhood": [["tag"]]}
    check.validate(strengthen)
    check.validate(update)
    assert not check.is_valid({**update, "actions": ["update_neighbor", "update_neighbor"]})
    assert not check.is_valid({**update, "should_evolve": False})
    empty = Draft202012Validator(helper_schema("memory_evolve", state=dict(neighbors=[])))
    empty.validate(noop)
    assert not empty.is_valid(strengthen) and not empty.is_valid(update)
    with pytest.raises(ValueError, match="current neighbors"):
        helper_schema("memory_evolve")


def test_amem_contract_rejects_repeated_tag_overflow_without_truncating_raw_notes():
    from agents.memory_baseline_control import AMEM_TEXT_LIMITS
    check = Draft202012Validator(helper_schema('memory_evolve', state=dict(neighbors=[dict(id='n1')])))
    decision = copy.deepcopy(EVOLUTION)
    check.validate(decision)
    # Regression for the Vista 1024-token loop in repeated neighborhood tags.
    bad = copy.deepcopy(decision)
    bad['new_tags_neighborhood'] = [['puzzle', 'film_criticism', 'biography', 'uncertain_data'] * 20]
    assert not check.is_valid(bad)
    for field, value in [('tags_to_update', ['tag'] * 5),
                         ('new_context_neighborhood', ['x' * 241]),
                         ('new_tags_neighborhood', [['x' * 33]])]:
        assert not check.is_valid({**decision, field: value})
    analyze = Draft202012Validator(helper_schema('memory_analyze'))
    analyze.validate(ANALYSIS)
    for field, value in [('keywords', ['word'] * 9), ('tags', ['tag'] * 5),
                         ('context', 'x' * 241), ('keywords', ['x' * 33])]:
        assert not analyze.is_valid({**ANALYSIS, field: value})
    store = memory.AMemMemory(fake_embed)
    raw = 'original observation ' * 300
    nid = store.add(raw, ANALYSIS)
    assert store.nodes[nid]['content'] == raw
    assert len(raw) > AMEM_TEXT_LIMITS['context_chars']


def test_evolution_request_uses_ordered_snapshot_and_rejects_misalignment_atomically(monkeypatch):
    # Second evolution has two neighbors, but the mocked server updates only one.
    responses = [SEARCH, json.dumps(ANALYSIS), OPEN, json.dumps(ANALYSIS),
                 json.dumps(EVOLUTION), SEARCH, json.dumps(ANALYSIS), json.dumps(EVOLUTION), FINISH]
    env, context, item, _ = setup(monkeypatch, "amem", responses)
    out = asyncio.run(runner.process_item(item, context))
    records = out.extra_fields["model_contexts"]
    evolutions = [r for r in records if r["phase"] == "memory_evolve"]
    assert len(evolutions) == 2
    for count, record in enumerate(evolutions, start=1):
        state = json.loads(record["messages"][1]["content"])
        assert len(state["neighbors"]) == count
        assert record["structured_outputs"] == {"json": helper_schema("memory_evolve", state=state)}
        ids = [node["id"] for node in state["neighbors"]]
        assert f"Neighbor order: {ids}" in record["messages"][0]["content"]
        assert f"exactly {count} entries" in record["messages"][0]["content"]
        assert record["control_enable_thinking"] is False
    snapshot = json.loads(evolutions[-1]["messages"][1]["content"])
    nodes = out.extra_fields["memory_state"]["nodes"]
    for node in [snapshot["new_note"], *snapshot["neighbors"]]:
        assert nodes[node["id"]] == node  # No partial strengthen before failed array update.
    stats = out.extra_fields["env_stats"]
    assert stats["invalid_memory"] == 1 and stats["amem_evolutions"] == 1
    assert stats["memory_requests"] == 5 and stats["model_requests"] == 9
    assert stats["main_len"] == sum(stats[k] for k in ("generated_tokens", "observation_tokens", "instruction_tokens"))
    assert out.extra_fields["termination_reason"] == "finish"
    assert out.extra_fields["baseline_protocol"]["evolution_output_profile"] == "neighbor_aligned_json_v1"
