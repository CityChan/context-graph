import asyncio
import sys
from types import SimpleNamespace

import pytest

from envs.judge_client import call_openai_raw


@pytest.mark.parametrize("outcome", ["ok", "failure", "cancel"])
def test_judge_client_closes_on_every_exit(monkeypatch, outcome):
    closed = []

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def create(self, **kwargs):
            if outcome == "failure":
                raise RuntimeError("unavailable")
            if outcome == "cancel":
                raise asyncio.CancelledError()
            assert kwargs["messages"] == [{"role": "user", "content": "test"}]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="yes"))])

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=Client))
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(call_openai_raw("test", max_retries=1))
    else:
        response = asyncio.run(call_openai_raw("test", max_retries=1))
        assert response == ("yes" if outcome == "ok" else "Error: unavailable")
    assert closed == [True]


@pytest.mark.parametrize("reply,expected", [("Error: quota", 0), ("unparseable", 0),
                                         ("<error> unrelated work", -1), ("<good>", 1)])
def test_scope_judge_does_not_penalize_transport_or_parse_failure(monkeypatch, reply, expected):
    # Load the small verifier without importing training/GPU dependencies.
    import importlib.util
    from pathlib import Path

    async def fake_call(*args, **kwargs):
        return reply

    monkeypatch.setitem(sys.modules, "agents.utils", SimpleNamespace(call_openai=fake_call))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    spec = importlib.util.spec_from_file_location("agents._verifier_test", Path(__file__).resolve().parents[1] / "agents/verifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    score, _ = asyncio.run(module.judge_scope("task", "result"))
    assert score == expected
