import asyncio
from types import SimpleNamespace

from agents.finalizer import (
    OBSERVATION_TRUNCATION_MARKER,
    append_observation_preserving_final_answer,
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
        self.emergency_finish_wrapped = False
        self.calls = []

    async def run_action(self, response):
        self.calls.append(response)
        if "<function=finish>" in response and "<parameter=answer>" in response:
            self.is_finish = True
            return {"action": "finish"}
        return {"observation": "not finished"}


class StructuredFakeAgent(FakeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.completion_kwargs = []

    async def step(self, max_new_tokens=None, completion_kwargs=None):
        self.step_budgets.append(max_new_tokens)
        self.completion_kwargs.append(completion_kwargs)
        return self.response


class OverlayCaptureAgent(FakeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_seen_during_step = None

    def replace_user_turn(self, idx, content):
        self._messages[idx]["content"] = content

    async def step(self, max_new_tokens=None, completion_kwargs=None):
        self.step_budgets.append(max_new_tokens)
        user_messages = [
            message["content"] for message in self._messages
            if message.get("role") == "user"
        ]
        self.content_seen_during_step = user_messages[-1]
        self._messages.append({"role": "assistant", "content": self.response})
        return self.response


async def fake_action_runner(env, response):
    result = await env.run_action(response)
    if result.get("action") == "finish":
        return None
    return result.get("observation")


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


class ObservationBudgetAgent:
    def __init__(self, context_len=60):
        self.prompt_ids_len = 20
        self.config = SimpleNamespace(response_length=80)
        self.tokenizer = WordTokenizer()
        self._context_len = context_len
        self._messages = [{"role": "user", "content": "question"}]
        self._costs = []

    def context(self):
        return list(range(self._context_len))

    def messages(self):
        return self._messages

    def append(self, turn):
        cost = len(self.tokenizer.encode(turn["content"])) + 2
        self._messages.append(turn)
        self._costs.append(cost)
        self._context_len += cost

    def rollback(self, k=1):
        for _ in range(k):
            self._messages.pop()
            self._context_len -= self._costs.pop()


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


def test_normal_step_forwards_structured_completion_constraints():
    agent = StructuredFakeAgent(context_len=60, response='{"summary":"ok"}')
    structured = {"structured_outputs": {"json": {"type": "object"}}}

    result = asyncio.run(step_preserving_final_answer(
        agent,
        reserve_tokens=20,
        completion_kwargs=structured,
    ))

    assert result == '{"summary":"ok"}'
    assert agent.step_budgets == [20]
    assert agent.completion_kwargs == [structured]


def test_normal_step_uses_memory_overlay_without_persisting_it():
    agent = OverlayCaptureAgent(context_len=40, response="normal")

    result = asyncio.run(step_preserving_final_answer(
        agent,
        reserve_tokens=0,
        context_overlay="[Current STRUCTMEM state] fact f1",
    ))

    assert result == "normal"
    assert "fact f1" in agent.content_seen_during_step
    assert agent.messages()[0]["content"] == "evidence"
    assert all("fact f1" not in message["content"] for message in agent.messages())


def test_observation_is_truncated_without_consuming_protected_budget():
    agent = ObservationBudgetAgent(context_len=60)
    observation = " ".join(f"token-{i}" for i in range(50))

    fitted = append_observation_preserving_final_answer(
        agent, observation, reserve_tokens=20, safety_tokens=2
    )

    assert fitted is not None
    assert fitted != observation
    assert OBSERVATION_TRUNCATION_MARKER.strip() in fitted
    assert remaining_generation_tokens(agent) >= 22
    assert agent.messages()[-1]["content"] == fitted


def test_observation_is_skipped_when_only_protected_budget_remains():
    agent = ObservationBudgetAgent(context_len=79)

    fitted = append_observation_preserving_final_answer(
        agent, "search result", reserve_tokens=20, safety_tokens=2
    )

    assert fitted is None
    assert len(agent.messages()) == 1
    assert remaining_generation_tokens(agent) == 21


def test_zero_reserve_keeps_legacy_full_observation_behavior():
    agent = ObservationBudgetAgent(context_len=60)

    fitted = append_observation_preserving_final_answer(
        agent, "full search result", reserve_tokens=0
    )

    assert fitted == "full search result"
    assert agent.messages()[-1]["content"] == "full search result"


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


def test_finalizer_uses_answer_graph_transiently_and_restores_prompt():
    response = "<function=finish><parameter=answer>Paris</parameter></function>"
    agent = OverlayCaptureAgent(context_len=40, response=response)
    env = FakeEnv()

    finished = asyncio.run(submit_emergency_final_answer(
        agent,
        env,
        reserve_tokens=80,
        action_runner=fake_action_runner,
        context_overlay="[STRUCTMEM evidence for final answer] Paris is the capital.",
    ))

    assert finished is True
    assert "STRUCTMEM evidence" in agent.content_seen_during_step
    assert "FINAL ANSWER REQUIRED NOW" in agent.content_seen_during_step
    assert "STRUCTMEM evidence" not in agent.messages()[0]["content"]
    assert "FINAL ANSWER REQUIRED NOW" in agent.messages()[0]["content"]


def test_finalizer_wraps_plain_text_as_finish_answer():
    agent = FakeAgent(context_len=40, response="Paris")
    env = FakeEnv()

    finished = asyncio.run(
        submit_emergency_final_answer(agent, env, reserve_tokens=80, action_runner=fake_action_runner)
    )

    assert finished is True
    assert "<parameter=answer>Paris</parameter>" in env.calls[0]
    assert env.emergency_finish_wrapped is True


def test_finalizer_preserves_native_finish_format_status():
    response = "<function=finish><parameter=answer>Paris</parameter></function>"
    agent = FakeAgent(context_len=40, response=response)
    env = FakeEnv()

    assert asyncio.run(
        submit_emergency_final_answer(agent, env, reserve_tokens=80, action_runner=fake_action_runner)
    )
    assert env.emergency_finish_wrapped is False


def test_finalizer_repairs_finish_call_with_missing_answer():
    agent = FakeAgent(context_len=40, response="<function=finish></function>")
    env = FakeEnv()

    finished = asyncio.run(
        submit_emergency_final_answer(agent, env, reserve_tokens=80, action_runner=fake_action_runner)
    )

    assert finished is True
    assert "<parameter=answer>Best effort:" in env.calls[0]
