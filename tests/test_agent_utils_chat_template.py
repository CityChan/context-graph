import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")

if importlib.util.find_spec("omegaconf") is None:
    omegaconf = types.ModuleType("omegaconf")
    omegaconf.DictConfig = object
    sys.modules["omegaconf"] = omegaconf

from agents.utils import AgentContext


class _UserRequiredTokenizer:
    """Minimal tokenizer matching templates that reject system-only chats."""

    _role_tokens = {
        "system": [10, 11],
        "user": [20, 21],
        "assistant": [30, 31],
    }

    def apply_chat_template(
        self, chat, *, add_generation_prompt=False, tokenize=True, **kwargs
    ):
        assert tokenize
        if not any(turn.get("role") == "user" for turn in chat):
            raise RuntimeError("No user query found in messages.")
        tokens = [
            token
            for turn in chat
            for token in self._role_tokens[turn["role"]]
        ]
        if add_generation_prompt:
            tokens.append(99)
        return tokens

    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())

    def decode(self, tokens, add_special_tokens=False):
        return "x " * len(tokens)


def test_agent_context_defers_unrenderable_system_only_prefix():
    config = SimpleNamespace(
        prompt_length=128,
        response_length=128,
        plugin=SimpleNamespace(),
    )
    chat = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "task"},
    ]

    context = AgentContext(chat, _UserRequiredTokenizer(), config, prompt_turn=2)

    assert context.chat_ids == [[], [10, 11, 20, 21]]
    assert context.prompt_ids_len == 4
    assert context.context() == [10, 11, 20, 21, 99]
