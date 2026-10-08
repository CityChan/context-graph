import asyncio
from types import SimpleNamespace

import pytest

from agents import supo_agent as supo
from scripts.eval_bcp_qwen38 import config_for
from tests.test_agentfold import SEARCH, OPEN, FINISH, setup as agentfold_setup


SUMMARY = '<summary>Verified gold in source [1]; inspect its details next.</summary>'


def setup(monkeypatch, responses, **kwargs):
    env, context, item, client = agentfold_setup(monkeypatch, responses, **kwargs)
    monkeypatch.setattr(supo, "select_env", lambda *a: lambda *a: env)
    monkeypatch.setattr(supo, "create_chat", lambda *a: [
        {"role": "system", "content": "Use search tools."},
        {"role": "user", "content": "ORIGINAL_QUESTION: Find gold."}])
    config = config_for(SimpleNamespace(method="supo", memory_mode="repaired"))
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.supo_context_threshold = 1700
    plugin.supo_summary_max_tokens = 500
    plugin.val_max_turn = 10
    context.config = config
    return env, context, item, client


def test_threshold_rollback_and_reset_change_actual_inputs_not_audit(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, SEARCH, SUMMARY, OPEN, FINISH])
    original_action = env.run_action

    async def tool(action):
        result = await original_action(action)
        if not env.is_finish:
            result["observation"] = ("DISCARDED_SECRET " * 100 if len(env.actions) == 2
                                     else "FIRST_EVIDENCE " * 50)
        return result

    env.run_action = tool
    out = asyncio.run(supo.process_item(item, context))
    assert env.actions == [SEARCH, SEARCH, OPEN, FINISH] and env.closed
    assert out.reward_score == 1
    records = out.extra_fields["model_contexts"]
    assert [r["phase"] for r in records] == ["action", "action", "summary", "action", "action"]
    summary_input = context.tokenizer.decode(records[2]["input_ids"])
    resumed_input = context.tokenizer.decode(records[3]["input_ids"])
    assert "FIRST_EVIDENCE" in summary_input and "DISCARDED_SECRET" not in summary_input
    assert "FIRST_EVIDENCE" not in resumed_input and "DISCARDED_SECRET" not in resumed_input
    assert "Verified gold in source [1]" in resumed_input and "ORIGINAL_QUESTION" in resumed_input
    assert "DISCARDED_SECRET" in records[1]["raw_observation"]
    assert records[1]["discarded_from_working_context"]
    assert [r["input_ids"] for r in records] == [ids for ids, _ in client.calls]
    assert out.prompt_ids == records[-1]["input_ids"] and out.response_ids == records[-1]["output_ids"]
    stats = out.extra_fields["env_stats"]
    assert stats["summary_restarts"] == stats["summary_attempts"] == 1
    assert stats["main_len"] == sum(stats[k] for k in ("generated_tokens", "observation_tokens", "instruction_tokens"))
    assert stats["generated_tokens"] == sum(len(r["output_ids"]) for r in records)
    assert not any(out.response_mask)


def test_multiple_resets_keep_original_question_and_latest_summary(monkeypatch):
    newer = '<summary>NEW_SUMMARY gold [1]</summary>'
    env, context, item, client = setup(monkeypatch, [SEARCH, SUMMARY, SEARCH, newer, FINISH],
                                      observation="large " * 400)
    out = asyncio.run(supo.process_item(item, context))
    assert out.extra_fields["env_stats"]["summary_restarts"] == 2
    last = context.tokenizer.decode(client.calls[-1][0])
    assert "NEW_SUMMARY" in last and "ORIGINAL_QUESTION" in last
    assert "Verified gold" not in last
    assert len(env.actions) == 3


def test_summary_limit_stops_without_extra_model_or_tool_calls(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, SUMMARY, SEARCH], observation="long " * 500)
    context.config.actor_rollout_ref.rollout.plugin.supo_max_summaries = 1
    out = asyncio.run(supo.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "summary_limit"
    assert out.extra_fields["env_stats"]["hit_summary_limit"] == 1
    assert not out.extra_fields["is_finish"]
    assert len(client.calls) == 3 and env.actions == [SEARCH, SEARCH] and env.closed


@pytest.mark.parametrize("bad", ["", "no summary tags", SEARCH, '<summary> </summary>',
                                  '<summary>a</summary><summary>b</summary>',
                                  '<summary>notes</summary>' + OPEN])
def test_bad_summary_never_dispatches_tools_or_replaces_history(monkeypatch, bad):
    env, context, item, client = setup(monkeypatch, [SEARCH, bad], observation="long " * 500)
    out = asyncio.run(supo.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "invalid_summary"
    assert out.extra_fields["env_stats"]["summary_restarts"] == 0
    assert env.actions == [SEARCH] and env.closed
    assert len(client.calls) == 2


def test_summary_request_counts_as_turn(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, SUMMARY], observation="long " * 500)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.val_max_turn, plugin.final_answer_reserve = 2, 0
    out = asyncio.run(supo.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "max_turn"
    assert out.extra_fields["env_stats"]["summary_restarts"] == 1
    assert len(client.calls) == out.num_turns == 2 and env.actions == [SEARCH]


def test_invalid_action_retries_without_tool_side_effects(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH + OPEN, SEARCH, FINISH], observation="small")
    out = asyncio.run(supo.process_item(item, context))
    assert env.actions == [SEARCH, FINISH]
    assert out.extra_fields["env_stats"]["invalid_tool"] == 1
    assert "exactly one" in context.tokenizer.decode(client.calls[1][0])


def test_cumulative_budget_is_not_refunded_by_summarization(monkeypatch):
    # Keep enough per-turn room to reach the total cap without a finalizer.
    env, context, item, client = setup(monkeypatch,
        [SEARCH, SUMMARY, SEARCH, SUMMARY, SEARCH, SUMMARY, SEARCH, SUMMARY], observation="long " * 500)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.supo_max_summaries = 20
    plugin.val_response_length = 1100
    plugin.final_answer_reserve = 0
    out = asyncio.run(supo.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "token_limit"
    assert out.extra_fields["env_stats"]["main_len"] <= 1100
    assert out.extra_fields["env_stats"]["summary_restarts"] >= 1
    assert len(client.calls) < 8


@pytest.mark.parametrize("failure", ["tool", "judge"])
def test_service_failures_propagate_and_close(monkeypatch, failure):
    env, context, item, _ = setup(monkeypatch, [SEARCH, FINISH], failure=failure, observation="small")
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(supo.process_item(item, context))
    assert env.closed


def test_training_and_unsafe_configuration_rejected(monkeypatch):
    env, context, item, client = setup(monkeypatch, [])
    context.is_train = True
    with pytest.raises(ValueError, match="joint RL"):
        asyncio.run(supo.process_item(item, context))
    context.is_train = False
    context.config.actor_rollout_ref.rollout.plugin.supo_context_threshold = 32700
    with pytest.raises(ValueError, match="insufficient"):
        asyncio.run(supo.process_item(item, context))
    context.config.actor_rollout_ref.rollout.plugin.supo_context_threshold = 10
    with pytest.raises(ValueError, match="initial prompt"):
        asyncio.run(supo.process_item(item, context))
    assert env.closed and not client.calls


@pytest.mark.parametrize("key,value", [("supo_context_threshold", 0), ("supo_max_summaries", -1),
    ("supo_summary_max_tokens", 0), ("supo_max_summaries", True)])
def test_invalid_options_fail_before_rollout(key, value):
    with pytest.raises(ValueError, match="Invalid"):
        config_for(SimpleNamespace(method="supo", memory_mode="repaired", **{key: value}))


def test_config_dispatch_preserves_old_method_defaults():
    from scripts.eval_gaia import _process_item_for_workflow
    new = config_for(SimpleNamespace(method="supo", memory_mode="repaired"))
    old = config_for(SimpleNamespace(method="react", memory_mode="repaired"))
    assert _process_item_for_workflow("search_supo") is supo.process_item
    plugin = new.actor_rollout_ref.rollout.plugin
    for key in supo.settings(plugin):
        assert key not in old.actor_rollout_ref.rollout.plugin
        del plugin[key]
    plugin.workflow = "search"
    assert new == old


def test_real_local_search_summary_then_finish(monkeypatch):
    import numpy as np
    from envs.local_search import AsyncSearchClient
    from tests.test_session_restart import Client, Tokenizer
    calls, closed = [], []

    async def post(self, path, payload):
        calls.append(path)
        return [{"docid": "1", "url": "local/1", "title": "Evidence", "text": "gold evidence " * 250}]

    original_close = AsyncSearchClient.close
    async def close(self):
        closed.append(self)
        await original_close(self)

    monkeypatch.setenv("LOCAL_SEARCH_URL", "http://localhost:9999")
    monkeypatch.setattr(AsyncSearchClient, "_post", post)
    monkeypatch.setattr(AsyncSearchClient, "close", close)
    monkeypatch.setattr(supo, "create_chat", lambda *a: [
        {"role": "system", "content": "Use search tools."}, {"role": "user", "content": "Find gold."}])
    config = config_for(SimpleNamespace(method="supo", memory_mode="repaired", supo_context_threshold=1000))
    item = SimpleNamespace(non_tensor_batch={"ability": np.array(["LocalSearch"]),
        "uid": "test", "gen_uid": "test-gen", "extra_info": np.array([{
            "query": "Find gold.", "answer": "gold", "reward_mode": "searchr1_em",
            "workflow": "search_supo"}], dtype=object)})
    client = Client([SEARCH, SUMMARY, FINISH])
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), is_train=False, llm_client=client)
    out = asyncio.run(supo.process_item(item, context))
    assert out.reward_score == 1 and out.extra_fields["is_finish"]
    assert calls == ["/search"] and closed
    assert out.extra_fields["env_stats"]["summary_restarts"] == 1
    assert out.extra_fields["judge_audit"]


def test_timeout_and_model_errors_close_environment(monkeypatch):
    for error in (RuntimeError("model failed"), asyncio.CancelledError()):
        env, context, item, client = setup(monkeypatch, [])
        async def fail(*args, **kwargs):
            raise error
        client.create_completion = fail
        with pytest.raises(type(error)):
            asyncio.run(supo.process_item(item, context))
        assert env.closed
    env, context, item, client = setup(monkeypatch, [])
    context.config.actor_rollout_ref.rollout.plugin.session_timeout = -1
    out = asyncio.run(supo.process_item(item, context))
    assert not client.calls and env.closed
    assert out.extra_fields["termination_reason"] == "timeout"


def test_short_budget_reserves_final_answer_after_summary(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, SUMMARY, FINISH], observation="long " * 500)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.turn_max_new_tokens = 500
    plugin.val_response_length = 1100
    plugin.final_answer_reserve = 250
    out = asyncio.run(supo.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "finish"
    assert [r["phase"] for r in out.extra_fields["model_contexts"]] == ["action", "summary", "final"]
    assert out.extra_fields["env_stats"]["main_len"] <= 1100
    assert env.actions == [SEARCH, FINISH]


def test_supo_batch_wrapper_dispatches_only_supo():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/eval_bcp_supo_qwen35_9b_4node.sbatch").read_text()
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --nodes=4" in script
    assert 'eval_bcp_qwen35_9b_4node_idev.sh" supo' in script
    assert "SAMPLES=${SAMPLES:-8}" in script
