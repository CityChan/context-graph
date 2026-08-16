import asyncio
from types import SimpleNamespace

from agents.finalizer import (
    remaining_generation_tokens,
    step_preserving_final_answer,
    submit_emergency_final_answer,
)


class FakeAgent:
    def __init__(self, context_len=60, response=None):
        self.prompt_ids_len = 20
        self.config = SimpleNamespace(response_length=80)
        self._context_len = context_len
        self._messages = [{"role": "user", "content": "evidence"}]
        self.response = response
        self.step_budgets = []

    def context(self):
        return list(range(self._context_len))

    def messages(self):
        return self._messages

    def replace_user_turn(self, idx, content):
        self._messages[idx]["content"] = content
        self._context_len += 5

    def append(self, turn):
        self._messages.append(turn)
        self._context_len += 5

    async def step(self, max_new_tokens=None):
        self.step_budgets.append(max_new_tokens)
        self._messages.append({"role": "assistant", "content": self.response})
        return self.response


class FakeEnv:
    def __init__(self):
        self.is_finish = False
        self.calls = []

    async def run_action(self, response):
        self.calls.append(response)
        if "<function=finish>" in response and "<parameter=answer>" in response:
            self.is_finish = True
            return {"action": "finish"}
        return {"observation": "not finished"}


async def fake_action_runner(env, response):
    result = await env.run_action(response)
    if result.get("action") == "finish":
        return None
    return result.get("observation")


def test_normal_step_preserves_configured_final_answer_reserve():
    agent = FakeAgent(context_len=60, response="normal")

    result = asyncio.run(step_preserving_final_answer(agent, reserve_tokens=20))

    assert result == "normal"
    assert remaining_generation_tokens(agent) == 40
    assert agent.step_budgets == [20]


def test_normal_step_stops_when_only_reserve_remains():
    agent = FakeAgent(context_len=85, response="normal")

    result = asyncio.run(step_preserving_final_answer(agent, reserve_tokens=20))

    assert result is None
    assert agent.step_budgets == []


def test_finalizer_submits_tool_call_and_uses_only_protected_budget():
    response = (
        "<function=finish>\n"
        "<parameter=answer>Paris</parameter>\n"
        "</function>"
    )
    agent = FakeAgent(context_len=60, response=response)
    env = FakeEnv()

    finished = asyncio.run(
        submit_emergency_final_answer(agent, env, reserve_tokens=80, action_runner=fake_action_runner)
    )

    assert finished is True
    assert env.is_finish is True
    assert agent.step_budgets == [35]
    assert "FINAL ANSWER REQUIRED NOW" in agent.messages()[-2]["content"]


def test_finalizer_wraps_plain_text_as_finish_answer():
    agent = FakeAgent(context_len=40, response="Paris")
    env = FakeEnv()

    finished = asyncio.run(
        submit_emergency_final_answer(agent, env, reserve_tokens=80, action_runner=fake_action_runner)
    )

    assert finished is True
    assert "<parameter=answer>Paris</parameter>" in env.calls[0]


def test_finalizer_repairs_finish_call_with_missing_answer():
    agent = FakeAgent(context_len=40, response="<function=finish></function>")
    env = FakeEnv()

    finished = asyncio.run(
        submit_emergency_final_answer(agent, env, reserve_tokens=80, action_runner=fake_action_runner)
    )

    assert finished is True
    assert "<parameter=answer>Best effort:" in env.calls[0]
