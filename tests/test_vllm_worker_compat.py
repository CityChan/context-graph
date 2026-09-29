import ast
import inspect
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import pytest

from verl.utils.vllm_worker_compat import execute_worker_method, worker_wrapper_kwargs


class LegacyWrapper:
    def __init__(self, vllm_config, rpc_rank=0):
        self.config = vllm_config
        self.rpc_rank = rpc_rank

    def execute_method(self, method, *args, **kwargs):
        return method, args, kwargs


class ModernWrapper:
    def __init__(self, rpc_rank=0, global_rank=None):
        self.rpc_rank = rpc_rank
        self.global_rank = rpc_rank if global_rank is None else global_rank

    def init_worker(self, all_kwargs):
        self.config = all_kwargs[self.rpc_rank]['vllm_config']
        self.worker = SimpleNamespace(init_device=lambda: 'wrong target',
                                      sample_tokens=lambda grammar: grammar)

    def init_device(self):
        return 'wrapper context'

    def __getattr__(self, name):
        return getattr(self.worker, name)


@pytest.mark.parametrize('cls', [LegacyWrapper, ModernWrapper])
def test_constructor_contract_keeps_config_and_replica_local_rank(cls):
    config = object()
    wrapper = cls(**worker_wrapper_kwargs(cls, config))
    if cls is ModernWrapper:
        wrapper.init_worker([{'vllm_config': config, 'rank': 2}])
        assert wrapper.global_rank == 0  # Indexes this replica's single KV config.
    assert wrapper.config is config
    assert wrapper.rpc_rank == 0


def test_unknown_constructor_requirement_fails_before_loading_weights():
    class Unknown:
        def __init__(self, required_new_arg):
            pass

    with pytest.raises(TypeError, match='required_new_arg'):
        worker_wrapper_kwargs(Unknown, object())


def test_legacy_rpc_is_preserved():
    assert execute_worker_method(LegacyWrapper(None), 'operation', 1, flag=True) == ('operation', (1,), {'flag': True})


def test_modern_dispatch_preserves_wrapper_hooks_and_serialized_callables():
    wrapper = ModernWrapper()
    wrapper.init_worker([{'vllm_config': 'configuration'}])
    assert execute_worker_method(wrapper, 'init_device') == 'wrapper context'
    assert execute_worker_method(wrapper, 'sample_tokens', 'grammar') == 'grammar'
    operation = cloudpickle.dumps(lambda self, value: (self.config, value))
    assert execute_worker_method(wrapper, operation, 7) == ('configuration', 7)
    with pytest.raises(AttributeError):
        execute_worker_method(wrapper, 'unknown_method')


@pytest.mark.parametrize('modern', [True, False])
@pytest.mark.parametrize('non_block', [True, False])
@pytest.mark.parametrize('fail', [True, False])
def test_executor_sampling_arguments_and_future_error_propagation(modern, non_block, fail):
    class ModernExecutor:
        def sample_tokens(self, grammar_output, non_block=False):
            pass

    class LegacyExecutor:
        def sample_tokens(self, scheduler_output, output, non_block=False):
            pass

    # Execute the actual signature-selected adapter, without importing CUDA/Ray.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'verl/workers/rollout/vllm_rollout/vllm_async_server.py').read_text(encoding='utf8'))
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and 'inspect.signature(Executor.sample_tokens)' in ast.unparse(node.test))
    namespace = {'inspect': inspect, 'Executor': ModernExecutor if modern else LegacyExecutor, 'Future': Future}
    exec(compile(ast.Module(body=[branch], type_ignores=[]), 'sampling-adapter', 'exec'), namespace)
    calls = []

    def rpc(method, args):
        calls.append((method, args))
        if fail:
            raise RuntimeError('worker failed')
        return ['sampled']

    args = ('grammar',) if modern else ('schedule', 'output')
    def invoke():
        result = namespace['sample_tokens'](SimpleNamespace(collective_rpc=rpc), *args, non_block=non_block)
        if non_block:
            assert isinstance(result, Future)
            return result.result()
        return result

    if fail:
        with pytest.raises(RuntimeError, match='worker failed'):
            invoke()
    else:
        assert invoke() == 'sampled'
    assert calls == [('sample_tokens', args)]
