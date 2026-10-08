import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents import agentfold_agent as af
from scripts.eval_bcp_qwen38 import config_for
from tests.test_session_restart import Client, Tokenizer


SEARCH = '<function=search><parameter=query>gold</parameter></function>'
OPEN = '<function=open_page><parameter=docid>1</parameter></function>'
FINISH = '<function=finish><parameter=answer>gold</parameter></function>'


def compress(start, end, summary="gold evidence [1]"):
    return '<compress>' + json.dumps({"compress_range": [start, end], "compress_text": summary}) + '</compress>\n'


def test_suffix_folds_preserve_prefix_and_cannot_split_summaries():
    steps = [af.Step(1, 2, "older", True), af.Step(3, 3, "raw")]
    snapshot = copy.deepcopy(steps)
    small = af.fold_suffix(steps, {"compress_range": [3, 3], "compress_text": "latest"})
    assert small[0] == steps[0] and small[1].compressed
    merged = af.fold_suffix(small, {"compress_range": [1, 3], "compress_text": "merged"})
    assert merged == [af.Step(1, 3, "merged", True)]
    assert steps == snapshot
    assert "[Compressed Step 1 to 3]" in merged[0].render()


@pytest.mark.parametrize("span,summary", [([2, 3], "x"), ([1, 2], "x"),
    ([1, 4], "x"), ([0, 3], "x"), ([True, 3], "x"), ([1., 3], "x"),
    ([3, 1], "x"), ([1], "x"), ([1, 3], ""), ([1, 3], None)])
def test_reject_invalid_ranges_and_summaries(span, summary):
    with pytest.raises(ValueError):
        af.fold_suffix([af.Step(1, 2, "a", True), af.Step(3, 3, "b")],
                       {"compress_range": span, "compress_text": summary})


def test_gap_and_duplicate_tools_rejected_before_dispatch():
    with pytest.raises(ValueError, match="contiguous"):
        af.fold_suffix([af.Step(1, 1, "a"), af.Step(3, 3, "b")],
                       {"compress_range": [1, 3], "compress_text": "x"})
    for text in [SEARCH + SEARCH, '<compress>{bad}</compress>' + SEARCH,
                 '<compress>unfinished' + SEARCH, SEARCH.replace('query', 'wrong'),
                 SEARCH.replace('</function>', '<function=search></function></function>')]:
        with pytest.raises(ValueError):
            af.parse_response(text, [])
    assert af.parse_response('<think>example ' + SEARCH + '</think>' + SEARCH, [])[0] == SEARCH
    with pytest.raises(ValueError, match="finish"):
        af.parse_response(SEARCH, [], finish_only=True)
    assert af.parse_response(FINISH, [af.Step(1, 1, "raw")])[0] == FINISH
    with pytest.raises(ValueError, match="ambiguous"):
        af.parse_response(SEARCH.replace("gold", "<answer>hijack</answer>"), [])


def setup(monkeypatch, responses, *, observation="RAW_ONLY " * 100, failure=None):
    class Env:
        def __init__(self, *args):
            self.actions, self.stats = [], {}
            self.is_finish, self.closed = False, False
            self.instance_info = {"problem_statement": "Find gold."}

        async def init_env(self, item):
            pass

        async def run_action(self, action):
            self.actions.append(action)
            if failure == "tool":
                raise RuntimeError("tool transport failed")
            self.is_finish = action == FINISH
            return {"action": "finish"} if self.is_finish else {"observation": observation}

        async def get_reward(self, *args):
            if failure == "judge":
                raise RuntimeError("judge failed")
            return "ok", float(self.is_finish), {}

        async def close(self):
            self.closed = True

    env = Env()
    monkeypatch.setattr(af, "select_env", lambda *a: lambda *a: env)
    monkeypatch.setattr(af, "create_chat", lambda *a: [
        {"role": "system", "content": "tools"}, {"role": "user", "content": "Find gold."}])
    config = config_for(SimpleNamespace(method="agentfold", memory_mode="repaired"))
    config.actor_rollout_ref.rollout.plugin.val_max_turn = 10
    client = Client(responses)
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), llm_client=client, is_train=False)
    item = SimpleNamespace(non_tensor_batch={"ability": np.array(["LocalSearch"])})
    return env, context, item, client


def test_loop_folds_actual_input_keeps_raw_audit_and_budget(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, compress(1, 1) + OPEN,
        compress(1, 2, "ONLY_MERGED") + SEARCH, FINISH])
    out = asyncio.run(af.process_item(item, context))
    assert out.reward_score == 1 and env.closed
    assert env.actions == [SEARCH, OPEN, SEARCH, FINISH]
    prompts = [context.tokenizer.decode(ids) for ids, _ in client.calls]
    assert "RAW_ONLY" in prompts[1]
    assert "[Compressed Step 1]" in prompts[2] and "gold evidence [1]" in prompts[2]
    assert "[Compressed Step 1 to 2]" in prompts[3] and "ONLY_MERGED" in prompts[3]
    assert "gold evidence [1]" not in prompts[3]
    assert "Find gold." in prompts[3]
    audit = out.extra_fields["model_contexts"]
    assert [row["input_ids"] for row in audit] == [ids for ids, _ in client.calls]
    assert "RAW_ONLY" in audit[0]["raw_observation"]
    stats = out.extra_fields["env_stats"]
    assert stats["agentfold_folds"] == 2 and stats["agentfold_deep_folds"] == 1
    assert stats["main_len"] == stats["generated_tokens"] + stats["observation_tokens"]
    assert stats["generated_tokens"] == sum(len(r["output_ids"]) for r in audit)
    assert out.prompt_ids == audit[-1]["input_ids"]
    assert out.response_ids == audit[-1]["output_ids"]
    assert not any(out.response_mask)  # Evaluation only.


def test_invalid_fold_is_charged_without_tool_or_memory_mutation(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, compress(99, 99) + OPEN,
        compress(1, 1) + OPEN, FINISH])
    out = asyncio.run(af.process_item(item, context))
    assert env.actions == [SEARCH, OPEN, FINISH]
    assert out.extra_fields["env_stats"]["invalid_tool"] == 1
    assert "Format correction" in client.calls[2][1]["messages"][-1]["content"]
    assert "[Step 1]" in client.calls[2][1]["messages"][1]["content"]
    assert client.calls[2][1]["messages"][-2]["role"] == "assistant"
    assert out.num_turns == 4


def test_long_observation_reserves_finish_and_keeps_full_raw_audit(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, FINISH], observation="x" * 50000)
    out = asyncio.run(af.process_item(item, context))
    stats = out.extra_fields["env_stats"]
    assert stats["observation_budget_truncations"] == 1
    assert stats["main_len"] <= 24576 and out.extra_fields["is_finish"]
    assert len(out.extra_fields["model_contexts"][0]["raw_observation"]) == 50000
    assert "truncated by response budget" in client.calls[1][1]["messages"][-1]["content"]
    assert "submit one finish" in client.calls[1][1]["messages"][-1]["content"]
    assert all(len(ids) + kw["max_new_tokens"] <= 32768 for ids, kw in client.calls)


@pytest.mark.parametrize("failure", ["tool", "judge"])
def test_service_errors_propagate_and_environment_closes(monkeypatch, failure):
    env, context, item, _ = setup(monkeypatch, [SEARCH, FINISH], failure=failure)
    with pytest.raises(RuntimeError, match="failed"):
        asyncio.run(af.process_item(item, context))
    assert env.closed


def test_training_rejected_and_budget_stops_are_not_finish(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH])
    context.is_train = True
    with pytest.raises(ValueError, match="evaluation-only"):
        asyncio.run(af.process_item(item, context))
    context.is_train = False
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.val_max_turn, plugin.final_answer_reserve = 1, 0
    out = asyncio.run(af.process_item(item, context))
    assert out.extra_fields["termination_reason"] == "max_turn"
    assert not out.extra_fields["is_finish"] and out.reward_score == 0


def test_zero_budget_never_calls_model(monkeypatch):
    env, context, item, client = setup(monkeypatch, [])
    context.config.actor_rollout_ref.rollout.plugin.val_response_length = 5
    out = asyncio.run(af.process_item(item, context))
    assert not client.calls and env.closed
    assert out.extra_fields["termination_reason"] == "token_limit"


def test_config_and_dispatch_do_not_alias_foldagent():
    from scripts.eval_gaia import _process_item_for_workflow
    af_config = config_for(SimpleNamespace(method="agentfold", memory_mode="repaired"))
    old = config_for(SimpleNamespace(method="foldagent", memory_mode="repaired"))
    assert _process_item_for_workflow("search_agentfold") is af.process_item
    af_plugin = af_config.actor_rollout_ref.rollout.plugin
    assert af_plugin.workflow == "search_agentfold"
    assert old.actor_rollout_ref.rollout.plugin.workflow == "search_branch"
    af_plugin.workflow = "search_branch"
    assert af_config == old


def test_real_local_search_environment_and_judge(monkeypatch):
    from envs.local_search import AsyncSearchClient
    calls, closed = [], []

    async def post(self, path, payload):
        calls.append(path)
        return [{"docid": "1", "url": "local/1", "title": "Evidence", "text": "gold evidence"}]

    original_close = AsyncSearchClient.close
    async def close(self):
        closed.append(self)
        await original_close(self)

    monkeypatch.setenv("LOCAL_SEARCH_URL", "http://localhost:9999")
    monkeypatch.setattr(AsyncSearchClient, "_post", post)
    monkeypatch.setattr(AsyncSearchClient, "close", close)
    config = config_for(SimpleNamespace(method="agentfold", memory_mode="repaired"))
    item = SimpleNamespace(non_tensor_batch={
        "ability": np.array(["LocalSearch"]), "uid": "test", "gen_uid": "test-gen",
        "extra_info": np.array([{"query": "Find gold.", "answer": "gold",
                                  "reward_mode": "searchr1_em", "workflow": "search_agentfold"}], dtype=object)})
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), is_train=False,
                              llm_client=Client([SEARCH, compress(1, 1) + OPEN, FINISH]))
    out = asyncio.run(af.process_item(item, context))
    assert out.reward_score == 1 and out.extra_fields["is_finish"]
    assert calls == ["/search", "/open"] and closed
    assert out.extra_fields["env_stats"]["search"] == 1
    assert out.extra_fields["judge_audit"]


def test_model_error_and_cancellation_close_environment(monkeypatch):
    for error in (RuntimeError("model service failed"), asyncio.CancelledError()):
        env, context, item, client = setup(monkeypatch, [])
        async def fail(*args, **kwargs):
            raise error
        client.create_completion = fail
        with pytest.raises(type(error)):
            asyncio.run(af.process_item(item, context))
        assert env.closed


def test_expired_session_and_no_completion_are_not_success(monkeypatch):
    env, context, item, client = setup(monkeypatch, [])
    context.config.actor_rollout_ref.rollout.plugin.session_timeout = -1
    out = asyncio.run(af.process_item(item, context))
    assert not client.calls and env.closed
    assert out.extra_fields["termination_reason"] == "timeout"
    env, context, item, client = setup(monkeypatch, [])
    async def no_completion(*args, **kwargs):
        return None
    client.create_completion = no_completion
    out = asyncio.run(af.process_item(item, context))
    assert not out.extra_fields["is_finish"] and env.closed
    assert out.extra_fields["termination_reason"] == "token_limit"
