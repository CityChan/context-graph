"""Native adapters with real environment wrappers and scripted model/simulator I/O."""
import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from agents import memory_baseline_agent as runner, graph_memory_baselines as memory
from envs.scienceworld_env import ScienceWorldEnv
from envs.discoveryworld_env import DiscoveryWorldEnv
from scripts.eval_agent_benchmarks import config_for, agent_for
from tests.test_session_restart import Client, Tokenizer
from tests.test_discoveryworld import FakeAPI, call
from tests.test_graph_memory_baselines import PATCH, ANALYSIS, NOOP, PAIR, fake_embed
from tests.test_native_memobrain_recall import assignment


def scripted(action, method):
    responses = []
    for step in range(1, 7):
        responses.append(action)
        if step == 6:
            break
        responses.append(json.dumps(PATCH if method == 'memobrain' else ANALYSIS))
        if method == 'memobrain' and step % 2 == 0:
            responses.append(json.dumps(dict(nodes={}) if step == 2 else assignment({'3': 1})))
        if method == 'amem' and step > 1:
            responses.append(json.dumps(dict(should_evolve=True, actions=['strengthen', 'update_neighbor'],
                suggested_connections=['n1'], tags_to_update=['linked'],
                new_context_neighborhood=['earlier observed state'] * (step - 1),
                new_tags_neighborhood=[['source']] * (step - 1))))
    return responses


@pytest.mark.parametrize('method', ['memobrain', 'amem'])
@pytest.mark.parametrize('kind', ['scienceworld', 'discoveryworld'])
@pytest.mark.parametrize('success', [True, False])
def test_native_state_and_grading_survive_memory_operations(monkeypatch, tmp_path, method, kind, success):
    tokenizer = Tokenizer()
    config = config_for(kind, method, 65536, 100,
                        prompt_profile='focus_v3' if kind == 'scienceworld' else 'discoveryworld_v1')
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.memobrain_recall_interval = 2
    plugin.final_answer_reserve = 1024  # Simulator must ignore inherited finish reserves.
    original = memory.AMemMemory
    monkeypatch.setattr(memory, 'AMemMemory', lambda: original(fake_embed))
    if kind == 'scienceworld':
        class Simulator:
            def __init__(self, **kwargs):
                self.calls, self.closed = [], False
            def load(self, **kwargs):
                assert kwargs['generateGoldPath'] is False
            def reset(self):
                return 'initial room', {'score': 0}
            def get_task_description(self):
                return 'Boil the requested substance.'
            def step(self, command):
                self.calls.append(command)
                count = len(self.calls)
                return f'CURRENT_STATE_{count}', 0, count == 6, {
                    'score': (100 if success else -100) if count == 6 else 0, 'moves': count}
            def close(self):
                self.closed = True
        monkeypatch.setitem(sys.modules, 'scienceworld', SimpleNamespace(ScienceWorldEnv=Simulator))
        ability = 'ScienceWorld@native'
        env = ScienceWorldEnv(config.actor_rollout_ref.rollout, tokenizer, ability)
        item = SimpleNamespace(non_tensor_batch={'ability': np.array([ability]), 'extra_info': np.array([
            dict(task_name='boil', variation_idx=21, tool_log=str(tmp_path / 'tools.jsonl'))], dtype=object)})
        action = '<function=action><parameter=command>look around</parameter></function>'
    else:
        api = FakeAPI()
        api.getTaskScorecard = lambda: [dict(scoreNormalized=1.0 if success else 0.0,
            completed=api.ticks == 6, completedSuccessfully=success and api.ticks == 6, secret='ORACLE_SECRET')]
        api.ui[0].renderJSON = lambda: dict(world_steps=api.ticks, latest_state=f'CURRENT_STATE_{api.ticks}', taskProgress=[])
        class Env(DiscoveryWorldEnv):
            async def init_env(self, item):
                self._env = api
                self._actions = {'MOVE_DIRECTION': {'args': ['arg1']}}
                self.instance_info = dict(problem_statement='Move north.', tool_log=str(tmp_path / 'tools.jsonl'),
                                          grading_log=str(tmp_path / 'scorecard.json'))
                self.closed = False
            def close(self):
                self.closed = True
        ability = 'DiscoveryWorld@native'
        env = Env(config.actor_rollout_ref.rollout, tokenizer, ability)
        item = SimpleNamespace(non_tensor_batch={'ability': np.array([ability])})
        action = call(dict(action='MOVE_DIRECTION', arg1='north'))
    monkeypatch.setattr(runner, 'select_env', lambda *args: lambda *args: env)
    context = SimpleNamespace(config=config, tokenizer=tokenizer, is_train=False,
                              llm_client=Client(scripted(action, method)))
    out, = asyncio.run(agent_for(method)(item, context))
    stats = out.extra_fields['env_stats']
    assert out.extra_fields['termination_reason'] == 'finish'
    assert stats['environment_steps'] == 6 and stats['completed'] == int(success)
    assert stats['environment_score'] == (100 if success else (-100 if kind == 'scienceworld' else 0))
    assert stats['invalid_memory'] == stats['invalid_tool'] == 0
    assert stats['memobrain_folds' if method == 'memobrain' else 'amem_evolutions'] > 0
    assert stats['main_len'] == sum(stats[k] for k in ('generated_tokens', 'observation_tokens', 'instruction_tokens'))
    records = out.extra_fields['model_contexts']
    assert 'ORACLE_SECRET' not in str(records)
    assert all(r['phase'] != 'final' for r in records)
    actions = [r for r in records if r['phase'] == 'action']
    assert len(actions) == 6 and 'CURRENT_STATE_5' in str(actions[-1]['messages'])
    assert 'name=search' not in str(actions[0]['messages'])
    assert out.extra_fields[method]['task_protocol'] == kind + '_v1'
    assert len((tmp_path / 'tools.jsonl').read_text().splitlines()) == (7 if kind == 'scienceworld' else 6)


@pytest.mark.parametrize('method', ['memobrain', 'amem'])
def test_simulator_turn_budget_never_synthesizes_finish(monkeypatch, method):
    from tests.test_graph_memory_baselines import setup
    action = '<function=action><parameter=command>look around</parameter></function>'
    env, context, item, _ = setup(monkeypatch, method, [action])
    monkeypatch.setattr(runner, 'create_chat', lambda *args, **kwargs: [dict(role='system', content='Native action task')])
    item.non_tensor_batch['ability'] = np.array(['ScienceWorld@native'])
    context.config.actor_rollout_ref.rollout.plugin.workflow = 'scienceworld_' + method
    context.config.actor_rollout_ref.rollout.plugin.val_max_turn = 1
    out = asyncio.run(runner.process_item(item, context))
    assert env.actions == [action] and env.closed
    assert out.extra_fields['termination_reason'] == 'max_turn'
    assert out.extra_fields['env_stats']['memory_requests'] == 0
    assert all(r['phase'] == 'action' for r in out.extra_fields['model_contexts'])
