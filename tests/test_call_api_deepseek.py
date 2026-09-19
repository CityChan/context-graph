import asyncio
import importlib.util
import sys
import types
from types import SimpleNamespace

import openai
import pytest

pytest.importorskip("transformers")

if importlib.util.find_spec("omegaconf") is None:
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = object
    sys.modules["omegaconf"] = omegaconf

from agents.utils import CallAPI


class _Completions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
            usage=None,
        )


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())


def _client(monkeypatch, reasoning_effort):
    completions = _Completions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kwargs: fake_client)
    config = SimpleNamespace(
        prompt_length=128,
        response_length=128,
        plugin=SimpleNamespace(
            turn_max_new_tokens=64,
            temperature=1.0,
            top_p=0.95,
            reasoning_effort=reasoning_effort,
        ),
    )
    return CallAPI("deepseek-ai/DeepSeek-V4-Flash-0731", _Tokenizer(), config), completions


def test_call_api_sends_deepseek_non_thinking_template_kwargs(monkeypatch):
    client, completions = _client(monkeypatch, "non-thinking")
    result = asyncio.run(client.create_completion([1, 2], messages=[{"role": "user", "content": "hi"}]))
    assert result["choices"][0]["message"]["content"] == "done"
    assert completions.calls[0]["temperature"] == 1.0
    assert completions.calls[0]["top_p"] == 0.95
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"thinking": False},
    }


def test_call_api_sends_qwen_non_thinking_template_kwargs(monkeypatch):
    client, completions = _client(monkeypatch, "non-thinking")
    client.model = "contextgraph-qwen3-8b-controller-sft"
    asyncio.run(client.create_completion(
        [1, 2], messages=[{"role": "user", "content": "hi"}]
    ))
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {
            "thinking": False,
            "enable_thinking": False,
        },
    }


def test_call_api_sends_deepseek_reasoning_effort(monkeypatch):
    client, completions = _client(monkeypatch, "high")
    asyncio.run(client.create_completion([1, 2], messages=[{"role": "user", "content": "hi"}]))
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"},
    }


def test_call_api_merges_structured_outputs_with_chat_template_kwargs(monkeypatch):
    client, completions = _client(monkeypatch, "non-thinking")
    schema = {"type": "object", "properties": {}}
    asyncio.run(client.create_completion(
        [1, 2],
        messages=[{"role": "user", "content": "merge"}],
        structured_outputs={"json": schema},
    ))
    assert completions.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"thinking": False},
        "structured_outputs": {"json": schema},
    }


def test_call_api_can_send_openai_json_schema_response_format(monkeypatch):
    client, completions = _client(monkeypatch, "non-thinking")
    client.config.plugin.api_structured_output_mode = "response_format"
    schema = {"type": "object", "properties": {}}
    asyncio.run(client.create_completion(
        [1, 2],
        messages=[{"role": "user", "content": "merge"}],
        structured_outputs={"json": schema},
    ))
    assert completions.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "contextgraph_controller",
            "schema": schema,
        },
    }
    assert "structured_outputs" not in completions.calls[0]["extra_body"]


def test_call_api_can_send_legacy_guided_json(monkeypatch):
    client, completions = _client(monkeypatch, "non-thinking")
    client.config.plugin.api_structured_output_mode = "guided_json"
    schema = {"type": "object", "properties": {}}
    asyncio.run(client.create_completion(
        [1, 2],
        messages=[{"role": "user", "content": "merge"}],
        structured_outputs={"json": schema},
    ))
    assert completions.calls[0]["extra_body"]["guided_json"] == schema
    assert "structured_outputs" not in completions.calls[0]["extra_body"]
