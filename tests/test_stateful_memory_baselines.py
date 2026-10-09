"""Stateful adapters: real environment action/scoring paths, fake simulator/model."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents import agentfold_agent as af, supo_agent as supo
from agents.baseline_task_protocol import task_chat
from agents.prompts import create_chat
from envs.discoveryworld_env import DiscoveryWorldEnv
from scripts.eval_agent_benchmarks import config_for, agent_for
from tests.test_agentfold import compress
from tests.test_session_restart import Client, Tokenizer
from tests.test_discoveryworld import FakeAPI, call

ACTION = call({'action': 'MOVE_DIRECTION', 'arg1': 'north'})


@pytest.mark.parametrize('method', ['agentfold', 'supo'])
def test_discoveryworld_real_actions_score_and_memory(method, monkeypatch, tmp_path):
    api = FakeAPI()
    api.getTaskScorecard = lambda: [{'scoreNormalized': .75, 'completed': api.ticks == 3,
                                     'completedSuccessfully': api.ticks == 3, 'secret': 'ORACLE_SECRET'}]
    api.ui[0].renderJSON = lambda: {'world_steps': api.ticks, 'latest_state': f'CURRENT_STATE_{api.ticks}',
                                  'visible': 'x' * 6000, 'taskProgress': []}
    class Env(DiscoveryWorldEnv):
        async def init_env(self, item):
            self._env = api
            self._actions = {'MOVE_DIRECTION': {'args': ['arg1']}}
            self.instance_info = {'problem_statement': 'Move north. Initial state 0.',
                                  'tool_log': str(tmp_path / 'tools.jsonl'),
                                  'grading_log': str(tmp_path / 'scorecard.json')}
            self.closed = False
        def close(self):
            self.closed = True
    config = config_for('discoveryworld', method, 65536, 20,
                        prompt_profile='discoveryworld_v1', observation_profile='compact_v1')
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.supo_context_threshold = 13000
    # Even a nonzero inherited reserve must never invent a simulator finish tool.
    plugin.final_answer_reserve = 1024
    tokenizer = Tokenizer()
    env = Env(config.actor_rollout_ref.rollout, tokenizer, 'DiscoveryWorld@real')
    module = af if method == 'agentfold' else supo
    monkeypatch.setattr(module, 'select_env', lambda *a: lambda *a: env)
    responses = ([ACTION, compress(1, 1, 'previous state') + ACTION, compress(1, 2) + ACTION]
                 if method == 'agentfold' else [ACTION, ACTION, '<summary>Earlier measurements retained.</summary>', ACTION])
    client = Client(responses)
    context = SimpleNamespace(config=config, tokenizer=tokenizer, llm_client=client, is_train=False)
    item = SimpleNamespace(non_tensor_batch={'ability': np.array(['DiscoveryWorld@real'])})
    outputs = asyncio.run(agent_for(method)(item, context))
    out, = outputs
    stats = out.extra_fields['env_stats']
    assert env.closed and env.is_finish and api.ticks == 3 and len(api.calls) == 3
    assert out.extra_fields['termination_reason'] == 'finish'
    assert stats['environment_score'] == 75 and stats['completed'] == 1
    assert 'ORACLE_SECRET' not in str(out.extra_fields['model_contexts'])
    assert 'ORACLE_SECRET' in (tmp_path / 'scorecard.json').read_text()
    records = out.extra_fields['model_contexts']
    assert all(r['phase'] != 'final' for r in records)
    if method == 'agentfold':
        assert stats['agentfold_folds'] == 2
    else:
        assert stats['summary_restarts'] == 1
        assert records[1]['retained_after_summary']
        assert 'CURRENT_STATE_2' in str(records[-1]['messages'])
        assert not stats['supo_discarded_rounds']
        assert stats['main_len'] == sum(stats[k] for k in ('generated_tokens', 'observation_tokens', 'instruction_tokens'))
    assert len((tmp_path / 'tools.jsonl').read_text().splitlines()) == 3


def test_environment_tools_are_strict_and_preserve_python_whitespace():
    code = "\nprint(1 < 2)\n"
    tool = '<function=python_exec><parameter=code>' + code + '</parameter></function>'
    assert af.parse_response(tool, [], kind='swe')[0] == tool
    assert af.parse_response(ACTION, [], kind='discoveryworld')[0] == ACTION
    for kind, bad in [('swe', ACTION), ('discoveryworld', tool),
                      ('discoveryworld', '<function=finish><parameter=answer>done</parameter></function>'),
                      ('discoveryworld', call([])), ('swe', tool + tool)]:
        with pytest.raises(ValueError):
            af.parse_response(bad, [], kind=kind)


@pytest.mark.parametrize('kind', ['swe', 'discoveryworld'])
def test_prompts_expose_only_native_tools(kind):
    chat = task_chat(kind, 'PUBLIC_TASK', None, 'agentfold', create_chat)
    assert 'PUBLIC_TASK' in str(chat)
    assert 'name=search' not in str(chat) and '<function=search>' not in str(chat)
    assert 'python_exec' in str(chat) if kind == 'swe' else 'chosen_dialog_option_int' in str(chat)


def test_scienceworld_uses_native_protocol_and_recorded_prompt_profile():
    plugin = config_for('scienceworld', 'supo', 65536, prompt_profile='focus_v2').actor_rollout_ref.rollout.plugin
    assert plugin.workflow == 'scienceworld_supo'
    assert plugin.baseline_task_protocol == 'scienceworld_v1'
    assert plugin.scienceworld_prompt_profile == 'focus_v2'
    assert plugin.final_answer_reserve == 0
