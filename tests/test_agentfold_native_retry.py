"""Native recovery preserves live graph boundaries and executable tool payloads."""
import asyncio
import re

import numpy as np
import pytest

from agents import agentfold_agent as af
from agents.prompts import create_chat
from tests.test_agentfold import setup, compress


ACTIONS = {
    'swe': '<function=python_exec><parameter=code>\nif 1 < 2:\n    print("ok")\n</parameter></function>',
    'scienceworld': '<function=action><parameter=command>look in drawer</parameter></function>',
    'discoveryworld': '<function=action><parameter=command>{"action":"MOVE_DIRECTION","arg1":"north"}</parameter></function>',
}


@pytest.mark.parametrize('kind', ACTIONS)
def test_native_retry_grammar_uses_current_suffix_and_native_tools(kind):
    steps = [af.Step(1, 3, 'old', True), af.Step(4, 4, 'new')]
    regex = af.format_retry_constraint(steps, kind)['regex']
    for start in (1, 4):
        response = compress(start, 4, 'Evidence "quoted"\nnext') + ACTIONS[kind]
        assert re.fullmatch(regex, response)
        action, folded = af.parse_response(response, steps, kind=kind)
        assert action == ACTIONS[kind] and folded[-1].start == start
    for bad in (ACTIONS[kind], compress(1, 3) + ACTIONS[kind], compress(2, 4) + ACTIONS[kind]):
        assert not re.fullmatch(regex, bad)
    finish = '<function=finish><parameter=message>done</parameter></function>'
    assert bool(re.fullmatch(regex, finish)) == (kind == 'swe')
    assert re.fullmatch(af.format_retry_constraint([], kind)['regex'], ACTIONS[kind])


@pytest.mark.parametrize('kind', ACTIONS)
@pytest.mark.parametrize('bad_fold', ['', '<compress>{"compress_range":[1,2],"compress_text":"unclosed</compress>'])
def test_native_recovery_does_not_replay_or_dispatch_invalid_actions(monkeypatch, kind, bad_fold):
    action = ACTIONS[kind]
    responses = [action, compress(1, 1) + action, bad_fold + action,
                 compress(2, 2, 'model summary') + action, compress(1, 3) + action]
    env, context, item, client = setup(monkeypatch, responses, observation='current state')
    monkeypatch.setattr(af, 'create_chat', create_chat)
    ability = {'swe': 'SWEVerified@real', 'scienceworld': 'ScienceWorld@real',
               'discoveryworld': 'DiscoveryWorld@real'}[kind]
    item.non_tensor_batch['ability'] = np.array([ability])
    original = env.run_action

    async def run(action):
        result = await original(action)
        env.is_finish = len(env.actions) == 4
        return result

    env.run_action = run
    out = asyncio.run(af.process_item(item, context))
    records = out.extra_fields['model_contexts']
    assert env.actions == [action] * 4 and env.closed
    assert out.extra_fields['termination_reason'] == 'finish'
    assert records[3]['phase'] == 'format_retry' and records[3]['control_enable_thinking'] is False
    assert records[4]['phase'] == 'action' and records[4]['structured_outputs'] is None
    assert 'model summary' in str(records[4]['messages'])
    assert out.extra_fields['env_stats']['agentfold_folds'] == 3
    assert out.extra_fields['agentfold']['variant'] == 'zero_shot_adaptation_v6'


@pytest.mark.parametrize('kind', ACTIONS)
def test_native_backend_ignoring_constraint_still_cannot_execute_bad_output(monkeypatch, kind):
    action = ACTIONS[kind]
    env, context, item, client = setup(monkeypatch, [action] * 4, observation='current state')
    monkeypatch.setattr(af, 'create_chat', create_chat)
    item.non_tensor_batch['ability'] = np.array([{
        'swe': 'SWEVerified@real', 'scienceworld': 'ScienceWorld@real',
        'discoveryworld': 'DiscoveryWorld@real'}[kind]])
    out = asyncio.run(af.process_item(item, context))
    assert env.actions == [action] and env.closed
    assert out.extra_fields['termination_reason'] == 'invalid_tool_limit'
    assert out.extra_fields['env_stats']['agentfold_folds'] == 0
    assert len(out.extra_fields['working_history']) == 1
