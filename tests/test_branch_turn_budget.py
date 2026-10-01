import asyncio
import collections
import importlib
from types import SimpleNamespace

import pytest

from tests.test_session_restart import Tokenizer, Client, make_agent


@pytest.mark.parametrize('budget', [0, 1, 2, 3])
def test_branch_summary_is_counted_inside_remaining_turn_budget(budget):
    async def run():
        agent = make_agent(['keep working'] * 10, enable_summary=False)
        async def action(response):
            return 'observation'
        result = await agent.react(action, max_turn=budget, summary_prompt='Return your findings')
        assert len(agent.llm_client.calls) <= budget
        assert result['iteration'] == len(agent.llm_client.calls)
        assert isinstance(result['last_response'], str)
    asyncio.run(run())


def test_branch_returns_early_and_does_not_summarize_after_timeout():
    async def run():
        async def action(response):
            pytest.fail('a return or an expired branch must not execute tools')
        agent = make_agent(['return'], enable_summary=False)
        result = await agent.react(action, max_turn=3, summary_prompt='summarize',
                                   should_continue=lambda response: response != 'return')
        assert result == {'last_response': 'return', 'iteration': 1}
        expired = make_agent([], enable_summary=False)
        result = await expired.react(action, max_turn=3, summary_prompt='summarize', session_timeout=-1)
        assert result['iteration'] == 0
        assert not expired.llm_client.calls
    asyncio.run(run())


@pytest.mark.parametrize('executor', [
    'fold_agent', 'fold_agent_code', 'graph_agent',
    'graph_agent_isolated', 'graph_agent_code_isolated'])
@pytest.mark.parametrize('budget', [1, 3])
@pytest.mark.parametrize('summary_enabled', [False, True])
def test_root_branch_and_checkpoint_share_turn_limit(executor, budget, summary_enabled, monkeypatch):
    from scripts.eval_swebench_verified import config_for, make_item

    module = importlib.import_module('agents.' + executor)
    class Env:
        def __init__(self, *args):
            self.stats = collections.Counter()
            self.instance_info = {'problem_statement': 'fix bug'}
            self.is_finish = False
        async def init_env(self, item):
            pass
        async def run_action(self, response):
            return {'observation': 'evidence'}
        async def get_reward(self, *args):
            return '', 0.0, {}
        def close(self):
            pass

    monkeypatch.setattr(module, 'select_env', lambda *a: Env)
    builder = 'create_chat_code' if 'code' in executor else 'create_chat'
    monkeypatch.setattr(module, builder, lambda *a, **kw: [
        {'role': 'system', 'content': 'tools'}, {'role': 'user', 'content': 'task'}])
    args = SimpleNamespace(method='foldagent', max_turn=budget, task_timeout=60, memory='8g', cpus=1)
    config = config_for(args)
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.structured_graph_controller = False
    plugin.consolidation_interval = 1
    plugin.enable_summary = summary_enabled
    plugin.summary_context_threshold = 1
    task = {'instance_id': 'test__test-1', 'repo': 'test/test', 'base_commit': 'a' * 40,
            'problem_statement': 'fix bug'}
    item = make_item(task, 'code_branch' if 'code' in executor else 'search')
    branch = '<function=branch><parameter=description>inspect</parameter><parameter=prompt>inspect code</parameter></function>'
    tool = '<function=python_exec><parameter=code>inspect()</parameter></function>'
    client = Client([branch] + [tool] * 15)
    context = SimpleNamespace(config=config, is_train=False, tokenizer=Tokenizer(), llm_client=client)
    asyncio.run(module.process_item(item, context))
    assert len(client.calls) <= budget
