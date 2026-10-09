"""Regressions for the full BC-P run's stale/missing/malformed folds."""
import asyncio
import re

import pytest

from agents import agentfold_agent as af
from tests.test_agentfold import setup, SEARCH, OPEN, FINISH, compress


def test_live_grammar_allows_model_choice_and_escaped_summary():
    steps = [af.Step(1, 3, 'old', True), af.Step(4, 4, 'new')]
    regex = af.search_retry_constraint(steps)['regex']
    for start in (1, 4):
        response = compress(start, 4, 'Evidence "quoted"\nsource \\path café') + OPEN
        assert re.fullmatch(regex, response)
        _, folded = af.parse_response(response, steps)
        assert folded[-1].start == start and folded[-1].end == 4
    assert re.fullmatch(regex, FINISH)
    for response in (compress(1, 3) + OPEN, compress(2, 4) + OPEN, OPEN,
                     compress(1, 4) + OPEN + SEARCH,
                     '<compress>{"compress_range":[1,4],"compress_text":"missing closure</compress>' + OPEN,
                     compress(1, 4, 'tag <function=search>') + OPEN):
        assert not re.fullmatch(regex, response)
    assert re.fullmatch(af.search_retry_constraint([])['regex'], SEARCH)
    assert not re.fullmatch(af.search_retry_constraint([])['regex'], compress(1, 1) + SEARCH)


@pytest.mark.parametrize('bad', [
    compress(1, 1) + OPEN,
    OPEN,
    '<compress>{"compress_range":[1,2],"compress_text":"unclosed</compress>' + OPEN,
])
def test_retry_constrains_current_state_then_restores_normal_decoding(monkeypatch, bad):
    env, context, item, client = setup(monkeypatch, [SEARCH, compress(1, 1) + OPEN,
        bad, compress(2, 2, 'fresh summary') + SEARCH, FINISH])
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.apply_chat_template_kwargs = dict(enable_thinking=True)
    renders = []
    original = context.tokenizer.apply_chat_template

    def render(messages, **kwargs):
        renders.append((messages, kwargs.copy()))
        return original(messages, **kwargs)

    monkeypatch.setattr(context.tokenizer, 'apply_chat_template', render)
    out = asyncio.run(af.process_item(item, context))
    records = out.extra_fields['model_contexts']
    assert env.actions == [SEARCH, OPEN, SEARCH, FINISH]
    assert [r['phase'] for r in records] == ['action', 'action', 'action', 'format_retry', 'action']
    retry = records[3]
    constraint = client.calls[3][1]['structured_outputs']
    assert constraint == retry['structured_outputs']
    assert re.fullmatch(constraint['regex'], compress(1, 2) + SEARCH)
    assert re.fullmatch(constraint['regex'], compress(2, 2) + SEARCH)
    assert not re.fullmatch(constraint['regex'], bad)
    assert '[Step 2]' in str(retry['messages'])
    assert 'fresh summary' in str(records[4]['messages'])
    for record in records:
        rendered = next(k for m, k in renders if m == record['messages'])
        assert rendered['enable_thinking'] is (record['phase'] != 'format_retry')
    assert 'structured_outputs' not in client.calls[4][1]
    assert plugin.apply_chat_template_kwargs.enable_thinking is True
    stats = out.extra_fields['env_stats']
    assert stats['invalid_tool'] == 1 and stats['agentfold_folds'] == 2
    assert stats['main_len'] == sum(stats[k] for k in
        ('generated_tokens', 'observation_tokens', 'feedback_tokens', 'instruction_tokens'))
    assert stats['main_len'] <= 24576 and env.closed


def test_ignored_retry_constraint_keeps_three_attempt_stop_and_memory(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, OPEN, OPEN, OPEN])
    out = asyncio.run(af.process_item(item, context))
    assert env.actions == [SEARCH] and env.closed
    assert out.extra_fields['termination_reason'] == 'invalid_tool_limit'
    assert out.extra_fields['env_stats']['agentfold_folds'] == 0
    assert out.extra_fields['env_stats']['invalid_tool'] == 3
    assert len(client.calls) == 4
    assert len(out.extra_fields['working_history']) == 1


def test_retry_backend_failure_surfaces_without_unconstrained_fallback(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, OPEN])
    original = client.create_completion

    async def serve(*args, **kwargs):
        if 'structured_outputs' in kwargs:
            raise RuntimeError('regex backend unavailable')
        return await original(*args, **kwargs)

    client.create_completion = serve
    with pytest.raises(RuntimeError, match='regex backend unavailable'):
        asyncio.run(af.process_item(item, context))
    assert env.actions == [SEARCH] and env.closed
