"""Real ScienceWorld adapter contract with a fake JVM simulator and model."""
import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from agents import agentfold_agent as af, supo_agent as supo
from agents.baseline_task_protocol import task_chat
from agents.prompts import create_chat
from agents.utils import _apply_chat_template
from envs.scienceworld_env import ScienceWorldEnv
from scripts.eval_agent_benchmarks import agent_for, config_for
from tests.test_agentfold import compress
from tests.test_session_restart import Client, Tokenizer

ACTION = '<function=action><parameter=command>look around</parameter></function>'


@pytest.mark.parametrize('method', ['supo', 'agentfold'])
@pytest.mark.parametrize('score', [100, -100])
@pytest.mark.parametrize('profile', ['focus_v2', 'focus_v3'])
def test_native_scienceworld_memory_and_terminal_scores(monkeypatch, tmp_path, method, score, profile):
    instances = []

    class Simulator:
        def __init__(self, **kwargs):
            self.calls, self.closed = [], False
            instances.append(self)
        def load(self, **kwargs):
            assert kwargs['generateGoldPath'] is False
        def reset(self):
            return 'room', {'score': 0}
        def get_task_description(self):
            return 'Boil the requested substance.'
        def step(self, command):
            self.calls.append(command)
            number = len(self.calls)
            return f'CURRENT_STATE_{number} ' + 'x' * 4000, 0, number == 3, {
                'score': score if number == 3 else 0, 'moves': number}
        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, 'scienceworld', SimpleNamespace(ScienceWorldEnv=Simulator))
    config = config_for('scienceworld', method, 65536, 100, prompt_profile=profile)
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.final_answer_reserve = 1024  # Must not introduce a finish tool.
    tokenizer = Tokenizer()
    task = SimpleNamespace(non_tensor_batch={
        'ability': np.array(['ScienceWorld@real']),
        'extra_info': np.array([{'task_name': 'boil', 'variation_idx': 21,
                                'tool_log': str(tmp_path / 'tools.jsonl')}], dtype=object)})
    fixed = task_chat('scienceworld', 'Boil the requested substance.\n\nInitial observation:\nroom',
                      task, method, create_chat, prompt_profile=profile)
    base = len(_apply_chat_template(tokenizer, fixed, config.actor_rollout_ref.rollout,
                                    tokenize=True, add_generation_prompt=True))
    plugin.supo_context_threshold = base + 6000
    responses = ([ACTION, ACTION, '<summary>Earlier state and requested goal.</summary>', ACTION]
                 if method == 'supo' else [ACTION, compress(1, 1) + ACTION, compress(1, 2) + ACTION])
    client = Client(responses)
    context = SimpleNamespace(config=config, tokenizer=tokenizer, llm_client=client, is_train=False)
    output, = asyncio.run(agent_for(method)(task, context))
    stats = output.extra_fields['env_stats']
    assert instances[0].calls == ['look around'] * 3 and instances[0].closed
    assert output.extra_fields['termination_reason'] == 'finish'
    assert stats['environment_score'] == score and stats['completed'] == int(score == 100)
    assert output.reward_score == int(score == 100)
    records = output.extra_fields['model_contexts']
    assert all(r['phase'] != 'final' for r in records)
    assert 'not an inspection or exploration command' in records[0]['messages'][0]['content']
    assert ('[Action guidance, not simulator observation]' in str(records[-1]['messages'])) == (profile == 'focus_v3')
    if method == 'supo':
        assert stats['summary_restarts'] == 1 and stats['supo_discarded_rounds'] == 0
        assert records[1]['retained_after_summary']
        assert 'CURRENT_STATE_2' in str(records[-1]['messages'])
    else:
        assert stats['agentfold_folds'] == 2
    assert output.extra_fields[method]['environment_finish'] is True
    rows = [json.loads(line) for line in (tmp_path / 'tools.jsonl').read_text().splitlines()]
    assert len(rows) == 4  # reset plus exactly three actions, never replayed.


def test_scienceworld_rejects_finish_and_multiline_commands_before_dispatch():
    assert af.parse_response(ACTION, [], kind='scienceworld')[0] == ACTION
    for response in [ACTION.replace('look around', 'look around\ninventory'),
                     ACTION.replace('look around', '"look around"'),
                     '<function=finish><parameter=answer>done</parameter></function>']:
        with pytest.raises(ValueError):
            af.parse_response(response, [], kind='scienceworld')


def test_scienceworld_simulator_failure_propagates_and_closes(monkeypatch):
    class Broken(ScienceWorldEnv):
        closed = False
        async def init_env(self, item):
            self.instance_info = {'problem_statement': 'goal'}
        async def run_action(self, response):
            self.env_fail = True
            raise RuntimeError('ScienceWorld simulator action failed')
        def close(self):
            self.closed = True

    config = config_for('scienceworld', 'supo', 65536, prompt_profile='focus_v2')
    env = Broken(config.actor_rollout_ref.rollout, Tokenizer(), 'ScienceWorld@real')
    monkeypatch.setattr(supo, 'select_env', lambda *args: lambda *args: env)
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), is_train=False,
                              llm_client=Client([ACTION]))
    task = SimpleNamespace(non_tensor_batch={'ability': np.array(['ScienceWorld@real'])})
    with pytest.raises(RuntimeError, match='simulator action failed'):
        asyncio.run(supo.process_item(task, context))
    assert env.closed
