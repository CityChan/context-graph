import asyncio
import collections
import importlib
from types import SimpleNamespace

import pytest

from agents.environment_lifecycle import managed_environment


def test_cleanup_waits_through_repeated_cancellation():
    async def run():
        entered, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        closed = []

        class Env:
            async def aclose(self):
                closing.set()
                await release.wait()
                closed.append(True)

        async def worker():
            async with managed_environment(Env()):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        await entered.wait()
        task.cancel()
        await closing.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [True]
    asyncio.run(run())


def test_cleanup_error_preserves_original_exception():
    class Env:
        def close(self):
            raise RuntimeError('close failed')

    async def run():
        async with managed_environment(Env()):
            raise ValueError('original')
    with pytest.raises(ValueError, match='original'):
        asyncio.run(run())


def test_cleanup_failure_on_normal_exit_is_visible():
    class Env:
        def close(self):
            raise RuntimeError('close failed')

    async def run():
        async with managed_environment(Env()):
            pass
    with pytest.raises(RuntimeError, match='close failed'):
        asyncio.run(run())


def test_search_connections_close_once_after_cancelled_reward_cleanup():
    from envs.local_search import AsyncSearchClient

    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        closed = []

        class Connection:
            async def aclose(self):
                started.set()
                await release.wait()
                closed.append(True)

        client = AsyncSearchClient.__new__(AsyncSearchClient)
        client._clients = [Connection()]
        client._close_task = None
        closing = asyncio.create_task(client.close())
        await started.wait()
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        release.set()
        await client.close()
        await client.close()
        assert closed == [True]
    asyncio.run(run())


def test_search_cleanup_awaits_other_connections_when_one_close_fails():
    from envs.local_search import AsyncSearchClient

    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        closed = []

        class Broken:
            async def aclose(self):
                raise RuntimeError('broken connection')

        class Healthy:
            async def aclose(self):
                started.set()
                await release.wait()
                closed.append(True)

        client = AsyncSearchClient.__new__(AsyncSearchClient)
        client._clients = [Broken(), Healthy()]
        client._close_task = None
        task = asyncio.create_task(client.close())
        await started.wait()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(RuntimeError, match='broken connection'):
            await task
        assert closed == [True]
    asyncio.run(run())


EXECUTORS = ['fold_agent', 'fold_agent_code', 'graph_agent', 'graph_agent_isolated',
             'graph_agent_code_isolated', 'react_agent', 'react_agent_code']


@pytest.mark.parametrize('executor', EXECUTORS)
@pytest.mark.parametrize('failure', ['init', 'model', 'cancel'])
def test_real_executors_close_environment_before_reward(executor, failure, monkeypatch):
    from scripts.eval_swebench_verified import config_for, make_item
    from tests.test_session_restart import Tokenizer

    module = importlib.import_module('agents.' + executor)
    instances = []

    class Env:
        def __init__(self, *args):
            self.stats = collections.Counter()
            self.instance_info = {'problem_statement': 'fix the bug'}
            self.closed = 0
            instances.append(self)

        async def init_env(self, item):
            if failure == 'init':
                raise ValueError('init failed')

        async def aclose(self):
            self.closed += 1

        async def get_reward(self, *args):
            pytest.fail('failure before reward must not trigger grading')

    class Client:
        async def create_completion(self, *args, **kwargs):
            if failure == 'cancel':
                raise asyncio.CancelledError
            raise ValueError('model failed')

    monkeypatch.setattr(module, 'select_env', lambda *args: Env)
    args = SimpleNamespace(method='foldagent', max_turn=5, task_timeout=60, memory='8g', cpus=1)
    config = config_for(args)
    config.actor_rollout_ref.rollout.plugin.structured_graph_controller = False
    task = {'instance_id': 'test__test-1', 'repo': 'test/test', 'base_commit': 'a' * 40,
            'problem_statement': 'fix the bug'}
    workflow = 'code_branch' if 'code' in executor else 'search'
    item = make_item(task, workflow)
    context = SimpleNamespace(config=config, is_train=False, tokenizer=Tokenizer(), llm_client=Client())
    error = asyncio.CancelledError if failure == 'cancel' else ValueError
    with pytest.raises(error):
        asyncio.run(module.process_item(item, context))
    assert len(instances) == 1 and instances[0].closed == 1


@pytest.mark.parametrize('executor', EXECUTORS[:5])
def test_real_executors_resume_summary_and_export_all_training_sessions(executor, monkeypatch):
    from scripts.eval_swebench_verified import config_for, make_item
    from tests.test_session_restart import Tokenizer, Client

    module = importlib.import_module('agents.' + executor)
    instances = []

    class Env:
        def __init__(self, *args):
            self.stats = collections.Counter()
            self.instance_info = {'problem_statement': 'fix the bug'}
            self.closed = 0
            self.is_finish = False
            instances.append(self)

        async def init_env(self, item):
            pass

        def close(self):
            self.closed += 1

        async def run_action(self, response):
            if '<function=finish>' in response:
                self.is_finish = True
                return {'action': 'finish'}
            return {'observation': 'old tool evidence ' * 650}

        async def get_reward(self, *args):
            return '', 1.0, {}

    monkeypatch.setattr(module, 'select_env', lambda *args: Env)
    args = SimpleNamespace(method='foldagent', max_turn=5, task_timeout=60, memory='8g', cpus=1)
    config = config_for(args)
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.enable_summary = True
    plugin.summary_context_threshold = 10000
    plugin.structured_graph_controller = False
    plugin.max_traj = 1
    plugin.consolidation_interval = 100
    prompt_builder = 'create_chat_code' if 'code' in executor else 'create_chat'
    monkeypatch.setattr(module, prompt_builder, lambda *a, **kw: [
        {'role': 'system', 'content': 'use tools'}, {'role': 'user', 'content': 'fix the bug'}])
    task = {'instance_id': 'test__test-1', 'repo': 'test/test', 'base_commit': 'a' * 40,
            'problem_statement': 'fix the bug'}
    workflow = 'code_branch' if 'code' in executor else 'search'
    item = make_item(task, workflow)
    client = Client(['<function=python_exec><parameter=code>inspect()</parameter></function>',
                     '<summary>fixed x.py; ready to finish</summary>',
                     '<function=finish><parameter=answer>done</parameter></function>'])
    context = SimpleNamespace(config=config, is_train=True, tokenizer=Tokenizer(), llm_client=client)
    outputs = asyncio.run(module.process_item(item, context))
    assert len(outputs) == 2
    assert all(x.reward_score == 1 for x in outputs)
    assert all(x.extra_fields['env_stats']['summary_restarts'] == 1 for x in outputs)
    assert 'fixed x.py' in str(client.calls[-1][1]['messages'])
    assert 'old tool evidence' not in str(client.calls[-1][1]['messages'])
    assert instances[0].closed == 1
