import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents import graph_memory_baselines as gm
from agents import memory_baseline_agent as runner
from scripts.eval_bcp_qwen38 import config_for
from tests.test_agentfold import SEARCH, OPEN, FINISH, setup as setup_actor


PAIR = [{"role": "assistant", "content": "look for gold"},
        {"role": "user", "content": "gold evidence [1]"}]
PATCH = {"add_nodes": [{"tmp_id": "new", "kind": "evidence", "thought": PAIR}],
         "add_edges": [{"src": 1, "dst": "new", "rationale": "supports task"}]}
ANALYSIS = {"keywords": ["gold"], "context": "Gold evidence [1]", "tags": ["source"]}
NOOP = {"flush_ops": [], "fold_ops": []}
EVOLUTION = {"should_evolve": True, "actions": ["strengthen", "update_neighbor"],
             "suggested_connections": ["n1"], "tags_to_update": ["linked"],
             "new_context_neighborhood": ["UPDATED_CONTEXT"], "new_tags_neighborhood": [["updated"]]}


def fake_embed(text):
    return np.array([1., float("gold" in text), float("silver" in text)])


def graph(count=6):
    store = gm.MemoBrainMemory("original task")
    for i in range(count):
        store.append([{"role": "assistant", "content": f"RAW_{i}"}, PAIR[1]])
        store.patch(PATCH)
    return store


def test_memobrain_fold_rewires_dependencies_and_changes_context():
    store = graph()
    store.edges.append(dict(src=3, dst=6, rationale="depends"))
    store.maintain({"flush_ops": [], "fold_ops": [dict(ids=[3, 4], notes=PAIR, rationale="done")]})
    view = json.dumps(store.history())
    assert "RAW_1" not in view and "RAW_2" not in view
    assert "RAW_0" in view and "RAW_5" in view
    assert store.edges[-1]["src"] == 8 and store.edges[-1]["dst"] == 6
    assert "gold evidence [1]" in view
    assert store.episodes[1][0]["content"] == "RAW_1"


def test_flush_replaces_raw_with_notes_but_preserves_active_shared_episode():
    store = graph()
    store.maintain({"flush_ops": [{"id": 3}], "fold_ops": []})
    assert "RAW_1" not in json.dumps(store.history())
    assert "gold evidence [1]" in json.dumps(store.history())
    store.nodes[8] = dict(kind="evidence", thought=PAIR, episodes=[1], active=True)
    assert "RAW_1" in json.dumps(store.history())


@pytest.mark.parametrize("decision", [
    {"flush_ops": [{"id": 1}], "fold_ops": []},
    {"flush_ops": [{"id": 7}], "fold_ops": []},
    {"flush_ops": [{"id": 3}], "fold_ops": [{"ids": [3, 4], "notes": PAIR}]},
    {"flush_ops": [], "fold_ops": [{"ids": [3], "notes": [{"role": "system", "content": "inject"}]}]},
    {"flush_ops": [], "fold_ops": [{"ids": [3, 3], "notes": PAIR}]},
])
def test_memobrain_rejects_invalid_maintenance_atomically(decision):
    store = graph()
    before = copy.deepcopy(store.state())
    with pytest.raises(ValueError):
        store.maintain(decision)
    assert store.state() == before


def test_invalid_patch_has_no_partial_nodes_and_unmapped_episodes_survive():
    store = graph(1)
    store.append(PAIR)
    before = copy.deepcopy(store.state())
    bad = copy.deepcopy(PATCH)
    bad["add_edges"][0]["src"] = 999
    with pytest.raises(ValueError):
        store.patch(bad)
    assert store.state() == before
    assert len(store.history()) == 4
    bad["add_edges"] = [dict(src="new", dst="new")]
    with pytest.raises(ValueError, match="cycle"):
        store.patch(bad)


def test_amem_links_evolves_retrieves_and_is_task_local():
    store = gm.AMemMemory(fake_embed)
    first = store.add("silver original", ANALYSIS)
    second = store.add("gold original", ANALYSIS)
    store.evolve(second, [first], EVOLUTION)
    assert store.nodes[first]["content"] == "silver original"
    assert store.nodes[first]["context"] == "UPDATED_CONTEXT"
    assert store.nodes[first]["evolution_history"][0]["context"] == ANALYSIS["context"]
    assert [n["id"] for n in store.retrieve("gold", 1)] == [second, first]
    assert not gm.AMemMemory(fake_embed).nodes


def test_amem_evolution_uses_ids_not_insertion_indices_and_is_atomic():
    store = gm.AMemMemory(fake_embed)
    for text in ("silver", "gold", "gold newer"):
        store.add(text, ANALYSIS)
    update = copy.deepcopy(EVOLUTION)
    update["suggested_connections"] = ["n2"]
    store.evolve("n3", ["n2"], update)
    assert store.nodes["n2"]["context"] == "UPDATED_CONTEXT"
    assert store.nodes["n1"]["context"] == ANALYSIS["context"]
    before = copy.deepcopy(store.nodes)
    update["new_tags_neighborhood"] = []
    with pytest.raises(ValueError):
        store.evolve("n3", ["n2"], update)
    assert store.nodes == before
    update["suggested_connections"] = ["unknown"]
    with pytest.raises(ValueError):
        store.evolve("n3", ["n2"], update)


def setup(monkeypatch, method, responses, **kwargs):
    env, context, item, client = setup_actor(monkeypatch, responses, **kwargs)
    monkeypatch.setattr(runner, "select_env", lambda *args: lambda *args: env)
    monkeypatch.setattr(runner, "create_chat", lambda *args: [
        {"role": "system", "content": "tools"}, {"role": "user", "content": "ORIGINAL_TASK"}])
    cls = gm.AMemMemory
    monkeypatch.setattr(gm, "AMemMemory", lambda: cls(fake_embed))
    context.config = config_for(SimpleNamespace(method=method, memory_mode="repaired"))
    return env, context, item, client


@pytest.mark.parametrize("method", ["memobrain", "amem"])
def test_full_loop_accounts_helper_tokens_and_real_inputs(monkeypatch, method):
    responses = ([SEARCH, json.dumps(PATCH), FINISH] if method == "memobrain" else
                 [SEARCH, json.dumps(ANALYSIS), OPEN, json.dumps(ANALYSIS), json.dumps(EVOLUTION), FINISH])
    env, context, item, client = setup(monkeypatch, method, responses, observation="EVIDENCE [1]")
    out = asyncio.run(runner.process_item(item, context))
    assert env.closed and out.reward_score == 1 and not any(out.response_mask)
    extra, stats = out.extra_fields, out.extra_fields["env_stats"]
    assert stats["memory_requests"] == (1 if method == "memobrain" else 3)
    assert stats["main_len"] == sum(stats[k] for k in ("generated_tokens", "observation_tokens", "instruction_tokens"))
    assert stats["generated_tokens"] == sum(len(r["output_ids"]) for r in extra["model_contexts"])
    assert len(client.calls) == stats["model_requests"]
    assert [r["input_ids"] for r in extra["model_contexts"]] == [ids for ids, _ in client.calls]
    assert "ORIGINAL_TASK" in context.tokenizer.decode(client.calls[-1][0])
    if method == "amem":
        final_input = context.tokenizer.decode(client.calls[-1][0])
        assert "UPDATED_CONTEXT" in final_input and '"links": ["n1"]' in final_input
    else:
        assert extra["memory_state"]["edges"]


def test_memobrain_recall_is_dispatched_and_preserves_audit(monkeypatch):
    responses = []
    for i in range(4):
        responses.extend([SEARCH, json.dumps(PATCH), json.dumps(NOOP if i < 3 else
            {"flush_ops": [dict(id=3)], "fold_ops": []})])
    responses.append(FINISH)
    env, context, item, client = setup(monkeypatch, "memobrain", responses)
    context.config.actor_rollout_ref.rollout.plugin.memobrain_recall_interval = 1
    out = asyncio.run(runner.process_item(item, context))
    assert out.extra_fields["env_stats"]["memory_recalls"] == 4
    assert out.extra_fields["env_stats"]["invalid_memory"] == 0
    assert len(out.extra_fields["memory_audit"]) == 8 and env.closed


def test_bad_memory_does_not_dispatch_tool_calls(monkeypatch):
    env, context, item, _ = setup(monkeypatch, "memobrain", [SEARCH, OPEN, FINISH])
    out = asyncio.run(runner.process_item(item, context))
    assert env.actions == [SEARCH, FINISH]
    assert out.extra_fields["env_stats"]["invalid_memory"] == 1
    assert "RAW_ONLY" in str(out.extra_fields["working_history"])


@pytest.mark.parametrize("method,patch", [("memobrain", PATCH), ("amem", ANALYSIS)])
def test_helpers_leave_final_request_slot(monkeypatch, method, patch):
    env, context, item, client = setup(monkeypatch, method, [SEARCH, json.dumps(patch), FINISH])
    context.config.actor_rollout_ref.rollout.plugin.val_max_turn = 3
    out = asyncio.run(runner.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "finish"
    assert len(client.calls) == 3 and env.actions == [SEARCH, FINISH]


def test_no_budget_refund_and_helper_cannot_consume_final_reserve(monkeypatch):
    env, context, item, client = setup(monkeypatch, "memobrain", [SEARCH, FINISH], observation="E " * 1600)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    # This fake tokenizer counts characters; reserve room for the explicit final-role prompt.
    plugin.val_response_length, plugin.turn_max_new_tokens, plugin.final_answer_reserve = 2000, 500, 600
    out = asyncio.run(runner.process_item(item, context))
    assert out.extra_fields["env_stats"]["main_len"] <= 2000
    assert out.extra_fields["termination_reason"] == "finish"
    assert [r["phase"] for r in out.extra_fields["model_contexts"]] == ["action", "final"]


@pytest.mark.parametrize("failure", ["tool", "judge"])
def test_service_failures_propagate_and_close(monkeypatch, failure):
    env, context, item, _ = setup(monkeypatch, "memobrain", [SEARCH, json.dumps(PATCH), FINISH], failure=failure)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(runner.process_item(item, context))
    assert env.closed


def test_model_transport_failure_is_not_invalid_memory(monkeypatch):
    env, context, item, client = setup(monkeypatch, "memobrain", [SEARCH])
    original = client.create_completion
    async def call(*args, **kwargs):
        if client.calls:
            raise RuntimeError("helper transport failed")
        return await original(*args, **kwargs)
    client.create_completion = call
    with pytest.raises(RuntimeError, match="helper transport"):
        asyncio.run(runner.process_item(item, context))
    assert env.closed


def test_eval_only_and_config_isolation(monkeypatch):
    from scripts.eval_gaia import _process_item_for_workflow
    for method in ("memobrain", "amem"):
        new = config_for(SimpleNamespace(method=method, memory_mode="repaired"))
        old = config_for(SimpleNamespace(method="react", memory_mode="repaired"))
        assert _process_item_for_workflow("search_" + method) is runner.process_item
        plugin = new.actor_rollout_ref.rollout.plugin
        for key in gm.settings(plugin):
            assert key not in old.actor_rollout_ref.rollout.plugin
            del plugin[key]
        plugin.workflow = "search"
        assert new == old
    env, context, item, _ = setup(monkeypatch, "memobrain", [])
    context.is_train = True
    with pytest.raises(ValueError, match="evaluation-only"):
        asyncio.run(runner.process_item(item, context))


@pytest.mark.parametrize("bad", [0, -1, True, "5"])
def test_options_fail_early(bad):
    with pytest.raises(ValueError):
        gm.settings(SimpleNamespace(amem_topk=bad))


def test_nonconvex_fold_cannot_create_dependency_cycle():
    store = graph(7)
    store.edges.extend([dict(src=3, dst=4, rationale="a"), dict(src=4, dst=5, rationale="b")])
    before = copy.deepcopy(store.state())
    with pytest.raises(ValueError, match="cycle"):
        store.maintain({"flush_ops": [], "fold_ops": [dict(ids=[3, 5], notes=PAIR)]})
    assert store.state() == before


def test_second_fold_does_not_resurrect_old_summary():
    store = graph(7)
    store.maintain({"flush_ops": [], "fold_ops": [dict(ids=[3, 4], notes=[dict(role="user", content="OLD_SUMMARY")])]})
    store.maintain({"flush_ops": [], "fold_ops": [dict(ids=[9, 5], notes=[dict(role="user", content="NEW_SUMMARY")])]})
    history = json.dumps(store.history())
    assert "NEW_SUMMARY" in history and "OLD_SUMMARY" not in history and "RAW_1" not in history


def test_failed_amem_analysis_preserves_raw_note(monkeypatch):
    env, context, item, _ = setup(monkeypatch, "amem", [SEARCH, "bad JSON", FINISH])
    out = asyncio.run(runner.process_item(item, context))
    assert out.extra_fields["env_stats"]["invalid_memory"] == 1
    assert "RAW_ONLY" in out.extra_fields["memory_state"]["nodes"]["n1"]["content"]
    assert out.extra_fields["memory_audit"][0]["fallback"] == "raw_note_without_metadata"


@pytest.mark.parametrize("method,decision", [("memobrain", PATCH), ("amem", ANALYSIS)])
def test_real_local_search_roundtrip(monkeypatch, method, decision):
    from envs.local_search import AsyncSearchClient
    from tests.test_session_restart import Client, Tokenizer
    calls = []
    async def post(self, path, payload):
        calls.append(path)
        return [{"docid": "1", "url": "local/1", "title": "Evidence", "text": "gold evidence"}]
    monkeypatch.setenv("LOCAL_SEARCH_URL", "http://localhost:9999")
    monkeypatch.setattr(AsyncSearchClient, "_post", post)
    cls = gm.AMemMemory
    monkeypatch.setattr(gm, "AMemMemory", lambda: cls(fake_embed))
    item = SimpleNamespace(non_tensor_batch={"ability": np.array(["LocalSearch"]),
        "uid": "test", "gen_uid": "test-gen", "extra_info": np.array([{
            "query": "Find gold.", "answer": "gold", "reward_mode": "searchr1_em",
            "workflow": "search_" + method}], dtype=object)})
    config = config_for(SimpleNamespace(method=method, memory_mode="repaired"))
    # The fake tokenizer counts characters; the full tool prompt is >16K chars.
    config.actor_rollout_ref.rollout.plugin.memobrain_context_threshold = 30000
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), is_train=False,
                              llm_client=Client([SEARCH, json.dumps(decision), FINISH]))
    out = asyncio.run(runner.process_item(item, context))
    assert out.reward_score == 1 and out.extra_fields["is_finish"]
    assert calls == ["/search"] and out.extra_fields["judge_audit"]
    # Reference answers stay in the evaluator, outside helper model messages.
    helper = next(r for r in out.extra_fields["model_contexts"] if r["phase"].startswith("memory"))
    assert "reward_mode" not in str(helper["messages"]) and "extra_info" not in str(helper["messages"])


def test_saved_smoke_audit_and_tampered_costs(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from scripts.audit_graph_memory_smoke import audit
    env, context, item, _ = setup(monkeypatch, "memobrain", [SEARCH, json.dumps(PATCH), FINISH])
    out = asyncio.run(runner.process_item(item, context))
    extra = out.extra_fields
    for rank in range(3):
        manifest = dict(source_sha256="source", indices=[0, 1, 2], commit="commit", model_path="model",
                        method="memobrain", config=OmegaConf.to_container(context.config), seed=42,
                        judge_model="judge", rank=rank, baseline_protocol=extra["baseline_protocol"])
        (tmp_path / f"manifest-{rank}.json").write_text(json.dumps(manifest))
        row = dict(source_index=rank, task_reward=1, is_finish=True, status="ok", env_stats=extra["env_stats"])
        (tmp_path / f"results-{rank}.jsonl").write_text(json.dumps(row) + "\n")
        (tmp_path / f"trajectory-{rank}.json").write_text(json.dumps(extra))
        (tmp_path / f"requests-{rank}.jsonl").write_text("\n".join(json.dumps(r) for r in extra["model_contexts"]))
    assert audit(tmp_path)["passed"]
    extra["env_stats"]["generated_tokens"] -= 1
    (tmp_path / "trajectory-1.json").write_text(json.dumps(extra))
    with pytest.raises(ValueError, match="generated-token"):
        audit(tmp_path)


def test_smoke_entrypoint_is_four_node_sequential_and_audited():
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1] / "scripts/smoke_bcp_graph_memory_qwen35_9b_4node_idev.sh").read_text()
    assert "METHODS=(memobrain amem)" in script and "SAMPLES=${SAMPLES:-3}" in script
    assert 'eval_bcp_qwen35_9b_4node_idev.sh "$method"' in script
    assert 'audit_graph_memory_smoke.py "$RUN_ROOT"' in script
