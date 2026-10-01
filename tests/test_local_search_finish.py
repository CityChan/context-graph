import asyncio
from collections import Counter
from types import SimpleNamespace

import pytest

from envs.local_search import LocalSearch


class Client:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True

    async def search(self, query, k):
        return []


def environment():
    env = LocalSearch.__new__(LocalSearch)
    env.stats = Counter()
    env.is_finish = False
    env.env_fail = False
    env.predicted_answer = None
    env.label_answer = "Paris"
    env.question = "Capital?"
    env.must_search = True
    env.donotgiveup = False
    env.double_check = False
    env.search_topk_cap = 5
    env.search_skill_context = None
    env.use_skills_only_memory = False
    env.client = Client()
    env.judge_audit = []
    return env


def finish(answer):
    return f"<function=finish><parameter=answer>{answer}</parameter></function>"


@pytest.mark.parametrize("answer", ["", "Paris", "London"])
def test_rejected_finish_does_not_set_completed_or_score_guess(answer):
    env = environment()

    async def check():
        for _ in range(2):
            result = await env.run_action(finish(answer))
            assert "observation" in result
            assert not env.is_finish
            assert not env.stats["is_finish"]
            assert env.predicted_answer is None
        reward = await env.get_reward(None, [{"role": "assistant", "content": "Answer: Paris"}], None)
        assert reward[1] == 0
        assert env.client.closed

    asyncio.run(check())


def test_valid_search_then_finish_accepts_answer():
    env = environment()

    async def score(answer, **kwargs):
        assert answer == "Paris"
        return 1

    env.score_answer = score

    async def check():
        await env.run_action("<function=search></function>")
        assert not env.stats["is_search"]
        assert "observation" in await env.run_action(finish("Paris"))
        await env.run_action("<function=search><parameter=query>capital France</parameter></function>")
        assert env.stats["is_search"]
        assert await env.run_action(finish("Paris")) == {"action": "finish"}
        assert env.is_finish
        assert (await env.get_reward(None, [], None))[1] == 1
        assert env.client.closed

    asyncio.run(check())


def test_double_check_keeps_proposal_unsubmitted():
    env = environment()
    env.stats["is_search"] = 1
    env.double_check = True
    assert "observation" in asyncio.run(env.run_action(finish("Paris")))
    assert not env.is_finish
    assert env.predicted_answer is None
    assert asyncio.run(env.get_reward(None, [], None))[1] == 0


def test_judge_exception_still_closes_search_client():
    env = environment()
    env.is_finish = True
    env.predicted_answer = ("Paris", "", 0)

    async def fail(*args, **kwargs):
        raise RuntimeError("judge unavailable")

    env.score_answer = fail
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(env.get_reward(None, [], None))
    assert env.client.closed
