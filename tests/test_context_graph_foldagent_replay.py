"""Replay real executors against a fixed model/tool tape (no GPU/network)."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("ray", reason="Real VERL executor imports require Ray")
from agents import fold_agent, graph_agent_isolated


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids if i)

    def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
        text = "".join(f"[{m['role']}]{m['content']}\0" for m in messages)
        return self.encode(text + ("[assistant]" if add_generation_prompt else ""))


def test_fixed_branch_return_replay_matches_foldagent(monkeypatch):
    class Env:
        def __init__(self, *args):
            self.stats = {}
            self.is_finish = False
            self.instance_info = {"problem_statement": "Find the access code."}

        async def init_env(self, item):
            pass

        async def get_reward(self, *args):
            return "", 1.0, {}

    async def action(env, response):
        if "<function=finish>" in response:
            env.is_finish = True
            return None
        return "The access code is 7319."

    monkeypatch.setattr(fold_agent, "select_env", lambda *args: Env)
    monkeypatch.setattr(fold_agent, "run_action", action)
    tape = [
        "<function=branch><parameter=description>verify</parameter><parameter=prompt>Find code</parameter></function>",
        "<function=search><parameter=query>code</parameter></function>",
        "<function=return><parameter=message>The code is 7319.</parameter></function>",
        "<function=finish><parameter=answer>7319</parameter></function>",
    ]

    async def run(equivalent):
        tokenizer = Tokenizer()
        responses = iter(tape)

        class Client:
            async def create_completion(self, *args, **kwargs):
                content = next(responses)
                ids = tokenizer.encode(content) + [0]
                return {"choices": [{"message": {"content": content, "raw_output_ids": ids, "response_log_probs": [0.0] * len(ids)}}]}

        plugin = SimpleNamespace(
            workflow="search_graph" if equivalent else "search_branch",
            contextgraph_memory_mode="foldagent" if equivalent else "legacy",
            controller_owned_tool_formatting=equivalent,
            capture_model_contexts=True, max_turn=10, max_session=3,
            process_reward=["flat"], session_timeout=60,
        )
        config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(
            plugin=plugin, prompt_length=65536, response_length=65536)))
        context = SimpleNamespace(config=config, is_train=False, tokenizer=tokenizer, llm_client=Client())
        item = SimpleNamespace(non_tensor_batch={
            "ability": np.array("search"), "uid": "replay", "gen_uid": "replay-generation",
            "extra_info": np.array({"workflow": plugin.workflow}, dtype=object),
        })
        runner = graph_agent_isolated.process_item if equivalent else fold_agent.process_item
        outputs = await runner(item, context)
        assert len(outputs) == 1
        assert len(outputs[0].extra_fields["branch_model_contexts"]) == 1
        return [{"requests": deepcopy(o.extra_fields["model_contexts"]), "branches": o.extra_fields["branch_model_contexts"], "messages": o.extra_fields["messages"],
                 "response_ids": o.response_ids, "reward": o.reward_score} for o in outputs]

    assert asyncio.run(run(False)) == asyncio.run(run(True))
