import importlib.util
import asyncio
import copy
import os
import sys
import types
from types import SimpleNamespace

import unittest
from unittest.mock import patch

if importlib.util.find_spec("transformers") is None:
    raise unittest.SkipTest("transformers is not installed")

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


class _ThinkingTokenizer(_UserRequiredTokenizer):
    def __init__(self):
        self.enable_thinking_values = []

    def apply_chat_template(self, chat, **kwargs):
        self.enable_thinking_values.append(kwargs.get("enable_thinking"))
        return super().apply_chat_template(chat, **kwargs)


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


def test_agent_context_chat_template_override_applies_to_every_render():
    config = SimpleNamespace(
        prompt_length=128,
        response_length=128,
        plugin=SimpleNamespace(),
    )
    tokenizer = _ThinkingTokenizer()
    chat = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "task"},
    ]

    with patch.dict(os.environ, QWEN_ENABLE_THINKING="True"):
        context = AgentContext(
            chat,
            tokenizer,
            config,
            prompt_turn=2,
            chat_template_kwargs={"enable_thinking": False},
        )
        context.context()

    assert tokenizer.enable_thinking_values
    assert set(tokenizer.enable_thinking_values) == {False}


class _RewritingTokenizer:
    """Qwen-style removal of reasoning before the last user message."""

    eos_token_id = 0

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, chat, add_generation_prompt=False, **kwargs):
        last_user = max((i for i, m in enumerate(chat) if m['role'] == 'user'), default=-1)
        parts = []
        for i, m in enumerate(chat):
            content = m['content']
            if m['role'] == 'assistant' and i < last_user and '</think>' in content:
                content = content.split('</think>', 1)[1]
            parts.append(f"[{m['role']}]{content}\0")
        return self.encode("".join(parts) + ("[assistant]" if add_generation_prompt else ""))


def _check_observation_roundtrip(tokenizer, observation, template_kwargs=None):
    config = SimpleNamespace(prompt_length=32768, response_length=32768, plugin=SimpleNamespace())
    chat = [{'role': 'system', 'content': 'Use tools.'}, {'role': 'user', 'content': 'Find evidence.'}]
    context = AgentContext(chat, tokenizer, config, chat_template_kwargs=template_kwargs)
    context.context()  # Cache the same generation prefix used for this completion.
    response = '<think>' + 'Reason about the evidence. ' * 150 + '</think>\n<function=search>query</function>'
    ids = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    completion = {'choices': [{'message': {'raw_output_ids': ids, 'response_log_probs': [-0.5] * len(ids)}}]}
    context.append({'role': 'assistant', 'content': response}, completion)
    preserved = copy.deepcopy((context.chat_ids, context.log_probs, context.token_mask))
    context.append({'role': 'user', 'content': observation})
    # Establish that this actually exercises the old broken prefix assumption.
    before = context._render_prefix(context.chat[:-1])
    after = context._render_prefix(context.chat)
    assert after[:len(before)] != before
    expected = context._render_prefix([context.chat[-1]])
    assert context.chat_ids[-1] == expected
    # Qwen3.5's template intentionally trims message-edge whitespace.
    assert observation.strip() in tokenizer.decode(context.context(), skip_special_tokens=False)
    assert (context.chat_ids[:-1], context.log_probs[:-1], context.token_mask[:-1]) == preserved
    assert not any(context.token_mask[-1])
    assert all(p == 0 for p in context.log_probs[-1])
    for tokens, probs, mask in zip(context.chat_ids, context.log_probs, context.token_mask):
        assert len(tokens) == len(probs) == len(mask)
    data = asyncio.run(context.get_data())
    assert len(data['response_ids']) == len(data['response_logprobs']) == len(data['response_mask'])
    assert [t for t, keep in zip(data['response_ids'], data['response_mask']) if keep] == ids
    context.replace_user_turn(3, 'Short evidence: 70%.')
    assert 'Short evidence: 70%.' in tokenizer.decode(context.context(), skip_special_tokens=False)
    assert (context.chat_ids[:-1], context.log_probs[:-1], context.token_mask[:-1]) == preserved
    context.rollback()
    assert (context.chat_ids, context.log_probs, context.token_mask) == preserved


OBSERVATIONS = [
    'Branch has finished its task: the reduction was 70% [70468].',
    'Search results: source 70468 reports a 70% reduction. ' * 120,
]


def test_rewritten_history_preserves_observation_and_training_alignment():
    for observation in OBSERVATIONS:
        _check_observation_roundtrip(_RewritingTokenizer(), observation)


def test_mapping_default_tokenizer_keeps_ids_and_generation_prefix():
    from transformers import BatchEncoding

    class MappingDefaultTokenizer(_RewritingTokenizer):
        def apply_chat_template(self, chat, return_dict=True, **kwargs):
            ids = super().apply_chat_template(chat, **kwargs)
            return BatchEncoding({'input_ids': ids, 'attention_mask': [1] * len(ids)}) if return_dict else ids

    tokenizer = MappingDefaultTokenizer()
    for observation in OBSERVATIONS:
        _check_observation_roundtrip(tokenizer, observation)
    chat = [{'role': 'system', 'content': 'Use tools.'}, {'role': 'user', 'content': 'Question'}]
    config = SimpleNamespace(prompt_length=32768, response_length=32768, plugin=SimpleNamespace())
    context = AgentContext(chat, tokenizer, config)
    expected = tokenizer.apply_chat_template(chat, return_dict=False, add_generation_prompt=True)
    assert context.context() == expected
    assert context.prompt_ids_len == len(tokenizer.apply_chat_template(chat, return_dict=False))


def test_context_dependent_standalone_turn_fails_instead_of_dropping_tokens():
    class UnsupportedTokenizer(_RewritingTokenizer):
        def apply_chat_template(self, chat, **kwargs):
            tokens = super().apply_chat_template(chat, **kwargs)
            return [999] + tokens if len(chat) == 1 else tokens

    with unittest.TestCase().assertRaisesRegex(ValueError, 'verified suffix'):
        _check_observation_roundtrip(UnsupportedTokenizer(), 'Branch evidence: 70%.')


def test_real_qwen3_observation_and_training_alignment():
    model = os.environ.get('QWEN_TOKENIZER_PATH')
    if not model:
        raise unittest.SkipTest('Set QWEN_TOKENIZER_PATH to a cached Qwen3 tokenizer for offline validation')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    for template_kwargs in ({}, {'enable_thinking': True, 'preserve_thinking': True}):
        for observation in OBSERVATIONS:
            _check_observation_roundtrip(tokenizer, observation, template_kwargs)


def load_tests(loader, tests, pattern):
    """Allow the offline cluster preflight to run without third-party pytest."""
    return unittest.TestSuite(
        unittest.FunctionTestCase(fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    )
