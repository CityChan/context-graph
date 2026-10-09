"""Versioned guidance must reach the model without changing simulator outcomes."""
import asyncio
import json

import pytest

from envs.scienceworld_env import ScienceWorldEnv
from envs.scienceworld_protocol import FOCUS_V3_REMINDER
from scripts.audit_stateful_memory_smoke import scienceworld_tail
from scripts.eval_agent_benchmarks import config_for


@pytest.mark.parametrize('profile', ['legacy', 'focus_v2', 'focus_v3'])
def test_focus_is_never_intercepted_or_rewritten(tmp_path, profile):
    calls = []

    class Simulator:
        def step(self, command):
            calls.append(command)
            if command == 'focus on tin':
                return 'No known action matches that input.', 0, False, {'score': 0}
            assert command == 'focus on drawer'
            return 'You focus on the drawer.', -100, True, {'score': -100}
        def close(self):
            pass

    config = config_for('scienceworld', 'supo', 65536, prompt_profile=profile)
    env = ScienceWorldEnv(config.actor_rollout_ref.rollout, None, 'ScienceWorld@real')
    env._env = Simulator()
    env.instance_info = {'tool_log': str(tmp_path / 'tools.jsonl')}
    action = '<function=action><parameter=command>{}</parameter></function>'
    first = asyncio.run(env.run_action(action.format('focus on tin')))
    assert first['observation'] == 'No known action matches that input.' + (
        '\n\n' + FOCUS_V3_REMINDER if profile == 'focus_v3' else '')
    assert not env.is_finish
    terminal = asyncio.run(env.run_action(action.format('focus on drawer')))
    assert calls == ['focus on tin', 'focus on drawer']
    assert terminal['action'] == 'finish' and env.is_finish and not env.env_fail
    assert FOCUS_V3_REMINDER not in terminal['observation']
    assert env.stats['environment_score'] == -100 and env.stats['completed'] == 0
    assert env.stats['environment_steps'] == 2
    raw = [json.loads(line) for line in (tmp_path / 'tools.jsonl').read_text().splitlines()]
    assert raw[0]['observation'] == 'No known action matches that input.'
    assert raw[-1]['reward'] == -100 and raw[-1]['done'] is True
    assert scienceworld_tail(tmp_path)[-1]['command'] == 'focus on drawer'
    env.close()


def test_compact_diagnostic_omits_large_valid_action_lists(tmp_path):
    rows = [{'event': 'reset'}] + [
        {'event': 'step', 'command': str(i), 'observation': 'x' * 1000,
         'info': {'score': i, 'valid': ['irrelevant action'] * 1000}, 'done': False}
        for i in range(10)]
    (tmp_path / 'tools.jsonl').write_text('\n'.join(map(json.dumps, rows)), encoding='utf8')
    tail = scienceworld_tail(tmp_path)
    assert [row['command'] for row in tail] == [str(i) for i in range(4, 10)]
    assert all(len(row['observation']) == 600 and 'valid' not in row for row in tail)
