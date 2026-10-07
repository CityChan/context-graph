import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents.graph_rpo_continuation import (
    Continuation, ReplayMismatch, bounded_candidates, choose_checkpoint, leave_one_out, recoverable,
)


def test_relative_credit_ties_and_partial_rewards():
    assert leave_one_out([1, 0, 1, 0]) == pytest.approx([2/3, -2/3, 2/3, -2/3])
    assert leave_one_out([0, 0, 0]) == [0, 0, 0]
    assert leave_one_out([1, 1]) == [0, 0]
    assert leave_one_out([0.2, 0.6]) == pytest.approx([-0.4, 0.4])
    with pytest.raises(ValueError):
        leave_one_out([float('nan'), 0])
    with pytest.raises(ValueError):
        leave_one_out([1])


def test_tape_is_strict_and_returns_independent_mutable_values():
    async def run():
        prefix = Continuation(None, 1)
        async def fetch():
            return {'pages': ['original']}
        result = await prefix.call('search', 'key', fetch)
        result['pages'].append('caller mutation')
        replay = Continuation(None, 1, prefix=prefix)
        result = await replay.call('search', 'key', fetch)
        assert result == {'pages': ['original']}
        result['pages'].clear()
        assert prefix.tape[0][2] == {'pages': ['original']}
        with pytest.raises(ReplayMismatch):
            await replay.call('search', 'extra', fetch)
        replay = Continuation(None, 1, prefix=prefix)
        with pytest.raises(ReplayMismatch):
            await replay.call('model', 'key', fetch)
    asyncio.run(run())


def _setup(monkeypatch, responses, *, checkpoint=1, train=True):
    from agents import graph_agent_isolated
    from envs.local_search import AsyncSearchClient
    from scripts.eval_discoverybench_qwen35 import config_for
    from tests.test_session_restart import Tokenizer, Client

    calls, closed = [], []
    async def post(self, path, payload):
        calls.append((path, copy.deepcopy(payload)))
        return [{'docid': '1', 'url': 'local/1', 'title': 'Evidence', 'text': 'gold evidence'}]
    original_close = AsyncSearchClient.close
    async def close(self):
        if self not in closed:
            closed.append(self)
        await original_close(self)
    monkeypatch.setattr(AsyncSearchClient, '_post', post)
    monkeypatch.setattr(AsyncSearchClient, 'close', close)
    monkeypatch.setenv('LOCAL_SEARCH_URL', 'http://localhost:9999')
    monkeypatch.setattr(graph_agent_isolated, 'create_chat', lambda *a, **kw: [
        {'role': 'system', 'content': 'Use tools and maintain the graph when requested.'},
        {'role': 'user', 'content': 'Find gold.'},
    ])
    config = config_for('contextgraph', 32768)
    config.algorithm.adv_estimator = 'graphrpo'
    config.algorithm.graphrpo_memory_only = True
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.workflow = 'search_graph'
    plugin.graph_rpo_credit_backend = 'old_policy_continuation'
    plugin.graph_rpo_continuation_samples = 2
    plugin.graph_rpo_continuation_checkpoint = checkpoint
    plugin.graph_rpo_continuation_concurrency = 1  # Scripted response order for serial regression cases.
    plugin.graph_controller_temperature = 0.8
    plugin.controller_allow_pass = True
    plugin.consolidation_interval = 2
    plugin.max_turn = plugin.val_max_turn = 8
    plugin.session_timeout = 60
    plugin.max_traj = 10
    item = SimpleNamespace(non_tensor_batch={
        'ability': np.array(['LocalSearch']), 'uid': 'question', 'gen_uid': 'source',
        'extra_info': np.array([{'query': 'Find gold.', 'answer': 'gold',
                                'reward_mode': 'searchr1_em', 'workflow': 'search_graph'}], dtype=object),
    })
    client = Client(responses)
    context = SimpleNamespace(config=config, is_train=train, global_step=3,
                              tokenizer=Tokenizer(), llm_client=client)
    return graph_agent_isolated, item, context, client, calls, closed


SEARCH = '<function=search><parameter=query>gold</parameter></function>'
OPEN = '<function=open_page><parameter=docid>1</parameter></function>'
def finish(answer):
    return f'<function=finish><parameter=answer>{answer}</parameter></function>'
def decision(action, indices):
    return json.dumps(dict(action=action, candidate_indices=indices, summary='', relation='semantic'))


def test_real_loop_replays_prefix_runs_tools_and_trains_only_selected_memory_turn(monkeypatch):
    module, item, context, client, calls, closed = _setup(monkeypatch, [
        SEARCH, SEARCH,
        decision('prune', [0]), OPEN, finish('wrong'),
        decision('pass', []), OPEN, finish('gold'),
    ])
    outputs = asyncio.run(module.process_item(item, context))
    assert len(outputs) == 2
    assert [o.reward_score for o in outputs] == [0, 1]
    assert len(client.calls) == 8  # No prefix model calls on replay.
    assert [path for path, _ in calls] == ['/search', '/search', '/open', '/open']
    assert len(closed) == 3  # Prefix and both independent continuation environments.
    assert client.calls[2][0] == client.calls[5][0]
    for index, output in enumerate(outputs):
        trained = ''.join(chr(t) for t, mask in zip(output.response_ids, output.response_mask) if mask).rstrip('\0')
        assert trained == (decision('prune', [0]) if index == 0 else decision('pass', []))
        assert output.response_logprobs[-2] == -0.125
        credit = output.extra_fields['graph_edit_credit_mask']
        assert {v for v, m in zip(credit, output.response_mask) if m} == {-1 if index == 0 else 1}
        assert all(v == 0 for v, m in zip(credit, output.response_mask) if not m)
        assert not any(output.extra_fields['process_reward_mask'])
        assert output.extra_fields['graph_decision_mask'] == output.response_mask
        audit = output.extra_fields['graph_rpo_continuation']
        assert audit['rewards'] == [0, 1]
        assert audit['global_step'] == 3
    assert outputs[0].extra_fields['uid'] == outputs[1].extra_fields['uid']
    assert outputs[0].extra_fields['gen_uid'] != outputs[1].extra_fields['gen_uid']
    assert outputs[0].extra_fields['graph_rpo_continuation']['state_hash'] == outputs[1].extra_fields['graph_rpo_continuation']['state_hash']


def test_no_checkpoint_does_not_train_executor(monkeypatch):
    module, item, context, client, calls, closed = _setup(monkeypatch, [SEARCH, finish('gold')])
    output, = asyncio.run(module.process_item(item, context))
    assert output.reward_score == 1
    assert not any(output.response_mask)
    assert output.extra_fields['graph_rpo_continuation']['skipped'] == 'checkpoint_not_reached'
    assert len(closed) == 1


def test_later_checkpoint_excludes_earlier_and_future_memory_decisions(monkeypatch):
    module, item, context, client, calls, closed = _setup(monkeypatch, [
        SEARCH, SEARCH, decision('pass', []), SEARCH, SEARCH,
        decision('prune', [0]), OPEN, finish('wrong'),
        decision('pass', []), OPEN, finish('gold'),
    ], checkpoint=2)
    outputs = asyncio.run(module.process_item(item, context))
    assert len(client.calls) == 11
    assert len(calls) == 6
    for output in outputs:
        assert output.extra_fields['graph_rpo_continuation']['checkpoint'] == 2
        text = ''.join(chr(t) for t, m in zip(output.response_ids, output.response_mask) if m).rstrip('\0')
        assert json.loads(text)['action'] in {'prune', 'pass'}
        assert output.extra_fields['graph_rpo_continuation']['rewards'] == [0, 1]


def test_branch_archive_is_reconstructed_without_training_branch_tokens(monkeypatch):
    branch = '<function=branch><parameter=description>subtask</parameter><parameter=prompt>Find evidence</parameter></function>'
    returned = '<function=return><parameter=message>Found evidence</parameter></function>'
    module, item, context, client, calls, closed = _setup(monkeypatch, [
        branch, SEARCH, returned, SEARCH,
        decision('pass', []), OPEN, finish('wrong'),
        decision('pass', []), OPEN, finish('gold'),
    ])
    outputs = asyncio.run(module.process_item(item, context))
    assert len(outputs) == 2 and len(client.calls) == 10
    assert len(calls) == 4 and len(closed) == 3
    assert outputs[0].extra_fields['graph_rpo_continuation']['rewards'] == [0, 1]
    assert outputs[0].extra_fields['num_branches'] == 1
    assert len({out.extra_fields['graph_rpo_continuation']['state_hash'] for out in outputs}) == 1


def test_failed_candidate_discards_group_and_closes_environment(monkeypatch):
    module, item, context, client, calls, closed = _setup(monkeypatch, [
        SEARCH, SEARCH, decision('pass', []), finish('gold'), decision('pass', []),
    ])
    create = client.create_completion
    async def failing(ids, **kwargs):
        if len(client.calls) == 5:
            raise RuntimeError('model unavailable')
        return await create(ids, **kwargs)
    client.create_completion = failing
    with pytest.raises(RuntimeError, match='model unavailable'):
        asyncio.run(module.process_item(item, context))
    assert len(closed) == 3


def test_checkpoint_state_mismatch_fails_closed(monkeypatch):
    from envs.local_search import LocalSearch
    module, item, context, client, calls, closed = _setup(monkeypatch, [SEARCH, SEARCH])
    init = LocalSearch.init_env
    count = 0
    async def changed(self, item):
        nonlocal count
        await init(self, item)
        self.replay_state_canary = count
        count += 1
    monkeypatch.setattr(LocalSearch, 'init_env', changed)
    with pytest.raises(ReplayMismatch, match='Maintenance state differs'):
        asyncio.run(module.process_item(item, context))
    assert len(client.calls) == 2 and len(closed) == 2


def test_judge_outage_is_not_a_zero_reward_training_label(monkeypatch):
    from envs.local_search import LocalSearch
    module, item, context, client, calls, closed = _setup(monkeypatch, [
        SEARCH, SEARCH, decision('pass', []), finish('gold'),
    ])
    async def unavailable(self, *args, **kwargs):
        raise RuntimeError('judge unavailable')
    monkeypatch.setattr(LocalSearch, 'score_answer', unavailable)
    with pytest.raises(RuntimeError, match='judge unavailable'):
        asyncio.run(module.process_item(item, context))
    assert len(closed) == 2


def test_inference_does_not_fork_or_change_masks(monkeypatch):
    responses = [SEARCH, SEARCH, decision('pass', []), finish('gold')]
    module, item, context, client, calls, closed = _setup(monkeypatch, responses, train=False)
    outputs = asyncio.run(module.process_item(item, context))
    assert len(outputs) == 1 and len(client.calls) == len(responses)
    assert len(closed) == 1
    assert 'graph_rpo_continuation' not in outputs[0].extra_fields
    assert sum(outputs[0].response_mask) > len(decision('pass', []))


def test_memory_advantages_do_not_include_episode_or_process_rewards():
    import torch
    from verl.trainer.ppo.core_algos import compute_graphrpo_advantage, compute_graphrpo_loss_weights

    mask = torch.tensor([[0., 1., 1.], [0., 0., 1.], [0., 0., 0.]])
    credit = torch.tensor([[0., -1., -1.], [0., 0., 1.], [0., 0., 0.]])
    kwargs = dict(token_level_rewards=torch.ones_like(mask), response_mask=mask,
                  index=np.array(['s', 's', 'skipped']), gen_uid=np.array(['a', 'b', 'c']),
                  process_reward_mask=-torch.ones_like(mask), graph_edit_credit_mask=credit,
                  graph_decision_mask=mask, config={'graphrpo_memory_only': True})
    advantages, _ = compute_graphrpo_advantage(**kwargs)
    assert torch.equal(advantages, credit)
    weights = compute_graphrpo_loss_weights(mask, kwargs['index'], kwargs['gen_uid'], allow_empty=True)
    assert weights.sum(dim=1).tolist() == pytest.approx([0.5, 0.5, 0])
    assert not compute_graphrpo_loss_weights(mask * 0, kwargs['index'], kwargs['gen_uid'], allow_empty=True).any()
    with pytest.raises(ValueError, match='only selected M'):
        compute_graphrpo_advantage(**{**kwargs, 'graph_decision_mask': torch.ones_like(mask)})
    # The driver pads by copying row 0 and zeroing only its response mask.
    padded_credit = credit.clone()
    padded_credit[2] = credit[0]
    padded_decisions = mask.clone()
    padded_decisions[2] = mask[0]
    advantages, _ = compute_graphrpo_advantage(**{
        **kwargs, 'graph_edit_credit_mask': padded_credit,
        'graph_decision_mask': padded_decisions, 'excluded_gen_uids': {'c'},
    })
    assert torch.equal(advantages, credit)


@pytest.mark.parametrize('alternating', [False, True])
def test_launcher_overrides_compose_into_training_config(tmp_path, alternating):
    import os
    import shutil
    import subprocess
    from pathlib import Path
    from hydra import compose, initialize_config_dir

    bash = shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    root = Path(__file__).resolve().parents[1]
    source = (root / 'scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh').read_text(encoding='utf8')
    block = source[source.index('GRAPH_RPO_ARGS=()'):source.index('# Qwen3-8B advertises')]
    script = tmp_path / 'overrides.sh'
    script.write_text('set -eu\n' + block + '\nprintf "%s\\n" "${GRAPH_RPO_ARGS[@]}" "$PROCESS_REWARD_SPEC"\n',
                      encoding='utf8', newline='\n')
    result = subprocess.run([bash, script.as_posix()], capture_output=True, text=True,
                            timeout=15, env={**os.environ, 'ADV_ESTIMATOR': 'graphrpo',
                                'BC_CTXGRAPH_PROTOCOL': 'controller',
                                'GRAPH_RPO_CREDIT_BACKEND': 'old_policy_continuation',
                                'GRAPH_RPO_ALTERNATING_ROLES': str(alternating),
                                'GRAPH_RPO_CONTINUATION_CHECKPOINT': 'random'},
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stderr
    *overrides, process_reward = result.stdout.splitlines()
    assert process_reward == '[]'
    with initialize_config_dir(config_dir=str(root / 'verl/trainer/config'), version_base=None):
        config = compose(config_name='ppo_trainer', overrides=overrides)
    assert config.algorithm.graphrpo_memory_only == (not alternating)
    assert config.algorithm.graphrpo_alternating_roles == alternating
    from verl.utils.config import omega_conf_to_dataclass
    assert omega_conf_to_dataclass(config.algorithm).graphrpo_alternating_roles == alternating
    assert config.algorithm.rollout_correction.rollout_is == 'token'
    assert config.actor_rollout_ref.rollout.plugin.graph_rpo_continuation_samples >= 2
    assert config.actor_rollout_ref.rollout.plugin.graph_rpo_continuation_checkpoint == 'random'
    from verl.utils.config import validate_config
    config.algorithm.adv_estimator = 'graphrpo'
    config.actor_rollout_ref.actor.policy_loss.loss_mode = 'graphrpo'
    config.actor_rollout_ref.actor.use_dynamic_bsz = True
    config.actor_rollout_ref.rollout.n = 1
    config.actor_rollout_ref.rollout.agent.default_agent_loop = 'context_graph_isolated_agent'
    from omegaconf import OmegaConf
    OmegaConf.update(config, 'actor_rollout_ref.rollout.plugin.structured_graph_controller', True, force_add=True)
    validate_config(config, use_reference_policy=False, use_critic=False)
    if alternating:
        config.actor_rollout_ref.rollout.plugin.graph_rpo_executor_samples = 1
        with pytest.raises(AssertionError):
            validate_config(config, use_reference_policy=False, use_critic=False)


def test_random_checkpoint_is_seeded_and_does_not_always_select_first():
    plugin = {'graph_rpo_continuation_checkpoint': 'random', 'graph_rpo_continuation_checkpoint_max': 4}
    draws = [choose_checkpoint(plugin, seed) for seed in range(64)]
    assert set(draws) == {1, 2, 3, 4}
    assert draws == [choose_checkpoint(plugin, seed) for seed in range(64)]
    assert choose_checkpoint({'graph_rpo_continuation_checkpoint': 3}, 99) == 3
    with pytest.raises(ValueError):
        choose_checkpoint({**plugin, 'graph_rpo_continuation_checkpoint_max': 0}, 1)


def test_bounded_candidates_overlap_preserve_order_and_drain_on_failure():
    async def run():
        active = peak = closed = 0
        both_started = asyncio.Event()
        async def candidate(index):
            nonlocal active, peak, closed
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
            try:
                await asyncio.wait_for(both_started.wait(), timeout=2)
                await asyncio.sleep(0)
                return index
            finally:
                active -= 1
                closed += 1
        assert await bounded_candidates(5, 2, candidate) == list(range(5))
        assert peak == 2 and active == 0 and closed == 5

        blocked = asyncio.Event()
        drained = asyncio.Event()
        async def failing(index):
            if index == 0:
                await blocked.wait()
                raise TimeoutError('service unavailable')
            blocked.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()
        with pytest.raises(TimeoutError):
            await bounded_candidates(2, 2, failing)
        assert drained.is_set()
    asyncio.run(run())


def test_caller_cancellation_is_propagated_after_cleanup():
    async def run():
        started, closed = asyncio.Event(), asyncio.Event()
        async def candidate(index):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        task = asyncio.create_task(bounded_candidates(2, 1, candidate))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
    asyncio.run(run())


def test_service_classification_does_not_hide_configuration_or_integrity_errors():
    import httpx
    import openai
    assert recoverable(TimeoutError())
    assert recoverable(httpx.ConnectError('connection reset'))
    assert not recoverable(httpx.UnsupportedProtocol('bad URL'))
    assert not recoverable(httpx.LocalProtocolError('invalid header'))
    for status in (400, 401, 403, 408, 429, 500, 503):
        response = httpx.Response(status, request=httpx.Request('POST', 'http://localhost'))
        error = httpx.HTTPStatusError('failed', request=response.request, response=response)
        assert recoverable(error) == (status in {408, 429} or status >= 500)
        error = openai.APIStatusError('failed', response=response, body=None)
        assert recoverable(error) == (status in {408, 429} or status >= 500)
    for error in (RuntimeError('bug'), ValueError('config'), ReplayMismatch('state')):
        assert not recoverable(error)


@pytest.mark.parametrize('stage', ['prefix_model', 'search', 'candidate', 'judge'])
def test_transient_failures_discard_entire_group_without_training_tokens(monkeypatch, stage):
    from envs.local_search import LocalSearch, AsyncSearchClient
    module, item, context, client, calls, closed = _setup(monkeypatch, [
        SEARCH, SEARCH, decision('pass', []), finish('gold'), decision('pass', []), finish('wrong'),
    ])
    async def unavailable(*args, **kwargs):
        raise TimeoutError('transient outage')
    if stage == 'judge':
        monkeypatch.setattr(LocalSearch, 'score_answer', unavailable)
    elif stage == 'search':
        monkeypatch.setattr(AsyncSearchClient, '_post', unavailable)
    else:
        create = client.create_completion
        async def flaky(ids, **kwargs):
            if len(client.calls) == (0 if stage == 'prefix_model' else 4):
                raise TimeoutError('transient outage')
            return await create(ids, **kwargs)
        client.create_completion = flaky
    output, = asyncio.run(module.process_item(item, context))
    assert not any(output.response_mask)
    assert not any(output.extra_fields['graph_edit_credit_mask'])
    assert output.extra_fields['graph_rpo_continuation']['skipped'] == 'service_failure'
    assert output.extra_fields['env_stats']['graph_rpo_continuation_failed'] == 1
    assert len(closed) == {'prefix_model': 1, 'search': 1, 'candidate': 3, 'judge': 2}[stage]


def test_real_loop_concurrent_candidates_have_independent_state(monkeypatch):
    from tests.test_session_restart import Client
    module, item, context, client, calls, closed = _setup(monkeypatch, [SEARCH, SEARCH])
    context.config.actor_rollout_ref.rollout.plugin.graph_rpo_continuation_concurrency = 2
    create = client.create_completion
    async def run():
        candidates = {}
        ready = asyncio.Event()
        async def concurrent(ids, **kwargs):
            if 'seed' not in kwargs.get('sampling_params', {}):
                return await create(ids, **kwargs)
            uid = kwargs['uid']
            if uid not in candidates:
                index = len(candidates)
                candidates[uid] = Client([decision('pass', []), OPEN, finish('gold' if index else 'wrong')])
                if len(candidates) == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=2)
            return await candidates[uid].create_completion(ids, **kwargs)
        client.create_completion = concurrent
        outputs = await module.process_item(item, context)
        assert [out.reward_score for out in outputs] == [0, 1]
        assert len(closed) == 3 and len(calls) == 4
        assert len({out.extra_fields['graph_rpo_continuation']['state_hash'] for out in outputs}) == 1
    asyncio.run(run())


def test_fatal_sibling_error_takes_precedence_over_service_failure():
    async def run():
        ready = asyncio.Event()
        async def candidate(index):
            if index == 0:
                await ready.wait()
                raise TimeoutError('service failure')
            ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                raise ReplayMismatch('integrity failure while closing')
        with pytest.raises(ReplayMismatch):
            await bounded_candidates(2, 2, candidate)
    asyncio.run(run())


@pytest.mark.parametrize('status', [401, 503])
def test_real_judge_path_preserves_error_types_for_continuations(monkeypatch, status):
    import httpx
    from envs import local_search
    module, item, context, client, calls, closed = _setup(monkeypatch, [
        SEARCH, SEARCH, decision('pass', []), finish('wrong'),
    ])
    item.non_tensor_batch['extra_info'][0]['reward_mode'] = 'default'
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    async def judge_failure(*args, **kwargs):
        assert kwargs['raise_errors'] is True
        response = httpx.Response(status, request=httpx.Request('POST', 'http://localhost'))
        raise httpx.HTTPStatusError('judge failure', request=response.request, response=response)
    monkeypatch.setattr(local_search, 'call_openai_raw', judge_failure)
    if status == 401:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(module.process_item(item, context))
    else:
        output, = asyncio.run(module.process_item(item, context))
        assert not any(output.response_mask)
        assert output.extra_fields['env_stats']['graph_rpo_continuation_failed'] == 1
    assert len(closed) == 2


@pytest.mark.parametrize('alternating', [False, True])
def test_driver_skips_empty_batches_without_entering_model_or_optimizer_workers(monkeypatch, alternating):
    import ast
    import uuid
    from pathlib import Path
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from verl import DataProto
    from verl.utils.debug import marked_timer
    from verl.utils import tracking

    # Execute the actual driver methods without importing optional GPU/PEFT engines.
    source = Path(__file__).resolve().parents[1] / 'verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(source.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {'fit', '_skip_empty_memory_batch'}]
    cls.decorator_list = []
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_configured_rollout_workflow')
    namespace = dict(torch=torch, uuid=uuid, DataProto=DataProto, tqdm=tqdm,
                     marked_timer=marked_timer)
    exec(compile(ast.Module(body=[helper, cls], type_ignores=[]), str(source), 'exec'), namespace)
    RayPPOTrainer = namespace['RayPPOTrainer']

    logs, audits, closed = [], [], []
    monkeypatch.setattr(tracking, 'Tracking', lambda **kw: SimpleNamespace(
        log=lambda **kw: logs.append(kw), finish=lambda: closed.append(True)))
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({
        'algorithm': {'graphrpo_memory_only': not alternating, 'graphrpo_alternating_roles': alternating},
        'trainer': {'project_name': 'test', 'experiment_name': 'test', 'logger': [],
                    'total_epochs': 1, 'rollout_data_dir': 'audit'},
        'global_profiler': {'steps': None, 'profile_continuous_steps': False},
        'actor_rollout_ref': {'rollout': {'temperature': 1., 'n': 1}},
    })
    trainer.train_dataloader = [{'input_ids': torch.ones(1, 2, dtype=torch.long)} for _ in range(2)]
    trainer.total_training_steps = 2
    trainer.train_dataset = SimpleNamespace()
    trainer._load_checkpoint = lambda: None
    trainer._start_profiling = lambda *args: None
    trainer._stop_profiling = lambda *args: None
    trainer.val_reward_fn = None
    trainer.async_rollout_mode = False
    trainer._log_rollout_data = lambda batch, *args: audits.append(batch)
    def generate(batch):
        result = DataProto.from_dict(tensors={'response_mask': torch.zeros(1, 2)},
                                     non_tensors={'env_stats': np.array([
                                         {'graph_rpo_continuation_failed': 1}], dtype=object)})
        result.meta_info['timing'] = {}
        return result
    # No model/scoring/update methods exist: entering any of them fails this test.
    trainer.actor_rollout_wg = SimpleNamespace(generate_sequences=generate)
    trainer.fit()
    assert len(logs) == 2 and len(audits) == 2 and closed == [True]
    assert all(row['data']['training/actor_update_calls'] == 0 for row in logs)
    assert all(row['data']['training/continuation_failed_groups'] == 1 for row in logs)
    assert trainer.global_steps == 3  # Bounded rollout attempts, no infinite resampling.
    if alternating:
        assert logs[0]['data']['graphrpo/E/actor_update_calls'] == 0
        assert logs[1]['data']['graphrpo/M/actor_update_calls'] == 0
    audits[0].batch['response_mask'][0, 0] = 1
    assert not trainer._skip_empty_memory_batch(audits[0], {}, {})  # A tied but valid group stays trainable.
    trainer.config.algorithm.graphrpo_memory_only = False
    trainer.config.algorithm.graphrpo_alternating_roles = False
    assert not trainer._skip_empty_memory_batch(audits[1], {}, {})  # Other backends unchanged.
