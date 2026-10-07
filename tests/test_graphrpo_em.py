"""Alternating shared-policy E/M: real agent loop, trainer credit and PPO gradients."""

import asyncio
import ast
from pathlib import Path
from typing import Any, Optional, Union
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tests.test_graph_rpo_continuation import _setup, SEARCH, OPEN, decision, finish
from tests.test_grpo_batch_wiring import _load_function
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.graph_rpo_roles import graph_rpo_update_role


def _em_setup(monkeypatch, responses, step=1, train=True):
    result = _setup(monkeypatch, responses, train=train)
    context = result[2]
    context.config.algorithm.graphrpo_memory_only = False
    context.config.algorithm.graphrpo_alternating_roles = True
    context.config.actor_rollout_ref.rollout.plugin.graph_rpo_executor_samples = 2
    context.global_step = step
    return result


def _trained(output):
    return ''.join(chr(t) for t, m in zip(output.response_ids, output.response_mask) if m).replace('\0', '')


def test_saved_step_selects_role_without_resetting_on_resume():
    config = {'graphrpo_alternating_roles': True}
    assert [graph_rpo_update_role(config, step) for step in (1, 2, 3, 18, 19)] == ['E', 'M', 'E', 'M', 'E']
    assert graph_rpo_update_role({}, 0) is None
    with pytest.raises(ValueError):
        graph_rpo_update_role(config, 0)
    with pytest.raises(ValueError):
        graph_rpo_update_role({**config, 'graphrpo_memory_only': True}, 1)


def test_registered_loop_receives_canonical_trainer_step():
    # Exercise the actual registration adapter; the driver sends global_steps (plural).
    source = Path(__file__).parents[1] / 'scripts/train_graph.py'
    tree = ast.parse(source.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ContextGraphIsolatedAgentLoop')
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'run')
    async def process(item, context):
        return graph_rpo_update_role(context.config.algorithm, context.global_step)
    namespace = dict(Any=Any, Union=Union, AgentLoopOutput=object, TaskContext=SimpleNamespace,
        DataProto=SimpleNamespace(from_dict=lambda **kw: None), CallLLM=lambda **kw: None,
        process_item_isolated=process)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    actor = SimpleNamespace(config=OmegaConf.create({'algorithm': {'graphrpo_alternating_roles': True},
        'actor_rollout_ref': {'rollout': {}}}), server_manager=None, tokenizer=None, loop=None)
    for step in (1, 2, 19, 20):
        assert asyncio.run(namespace['run'](actor, {}, global_steps=step, global_step=0)) == ('E' if step % 2 else 'M')


def test_real_executor_episodes_train_execution_and_exclude_all_memory_turns(monkeypatch):
    module, item, context, client, calls, closed = _em_setup(monkeypatch, [
        SEARCH, SEARCH, decision('pass', []), OPEN, finish('wrong'),
        SEARCH, SEARCH, decision('prune', [0]), OPEN, finish('gold'),
    ])
    outputs = asyncio.run(module.process_item(item, context))
    assert len(outputs) == 2 and len(closed) == 2
    assert len(client.calls) == 10 and len(calls) == 6  # Two whole episodes, no replay.
    for index, output in enumerate(outputs):
        assert output.extra_fields['graph_rpo_training_role'] == 'E'
        assert _trained(output) == SEARCH + SEARCH + OPEN + finish('gold' if index else 'wrong')
        assert any(output.extra_fields['graph_decision_mask'])
        assert not any(m and d for m, d in zip(output.response_mask, output.extra_fields['graph_decision_mask']))
        assert not any(output.extra_fields['graph_edit_credit_mask'])
        assert not any(output.extra_fields['process_reward_mask'])
        assert output.extra_fields['graph_rpo_continuation']['advantage'] == (1 if index else -1)
    assert outputs[0].extra_fields['uid'] == outputs[1].extra_fields['uid']
    assert outputs[0].extra_fields['gen_uid'] != outputs[1].extra_fields['gen_uid']


def test_executor_trains_branch_and_main_with_one_episode_identity(monkeypatch):
    branch = '<function=branch><parameter=description>subtask</parameter><parameter=prompt>Find evidence</parameter></function>'
    returned = '<function=return><parameter=message>Found evidence</parameter></function>'
    module, item, context, *_ = _em_setup(monkeypatch, [
        branch, SEARCH, returned, finish('wrong'), SEARCH, finish('gold'),
    ])
    outputs = asyncio.run(module.process_item(item, context))
    assert len(outputs) == 3
    branches = [o for o in outputs if o.extra_fields['agent_name'] != 'main']
    assert len(branches) == 1 and _trained(branches[0]) == SEARCH + returned
    matching = [o for o in outputs if o.extra_fields['gen_uid'] == branches[0].extra_fields['gen_uid']]
    assert len(matching) == 2 and {o.reward_score for o in matching} == {0}


def test_even_step_uses_same_state_memory_credit_and_masks_executor(monkeypatch):
    module, item, context, client, _, closed = _em_setup(monkeypatch, [
        SEARCH, SEARCH, decision('prune', [0]), finish('wrong'),
        decision('pass', []), finish('gold'),
    ], step=2)
    outputs = asyncio.run(module.process_item(item, context))
    assert len(client.calls) == 6 and len(closed) == 3
    assert [_trained(o) for o in outputs] == [decision('prune', [0]), decision('pass', [])]
    assert {o.extra_fields['graph_rpo_training_role'] for o in outputs} == {'M'}
    assert len({o.extra_fields['graph_rpo_continuation']['state_hash'] for o in outputs}) == 1


def test_em_validation_remains_one_ordinary_episode(monkeypatch):
    module, item, context, client, *_ = _em_setup(monkeypatch, [SEARCH, finish('gold')], train=False)
    outputs = asyncio.run(module.process_item(item, context))
    assert len(outputs) == 1 and len(client.calls) == 2
    assert 'graph_rpo_training_role' not in outputs[0].extra_fields
    assert _trained(outputs[0]) == SEARCH + finish('gold')


def test_executor_service_failure_discards_whole_group(monkeypatch):
    module, item, context, client, _, closed = _em_setup(monkeypatch, [SEARCH, finish('gold')])
    original = client.create_completion
    async def failing(*args, **kwargs):
        if len(client.calls) >= 2:
            raise TimeoutError('service unavailable')
        return await original(*args, **kwargs)
    client.create_completion = failing
    outputs = asyncio.run(module.process_item(item, context))
    assert len(outputs) == 1 and len(closed) == 2
    assert not any(outputs[0].response_mask)
    assert outputs[0].extra_fields['graph_rpo_training_role'] == 'E'
    assert outputs[0].extra_fields['graph_rpo_continuation']['skipped'] == 'service_failure'


def _compute(batch):
    function = _load_function(Path(__file__).parents[1] / 'verl/trainer/ppo/ray_trainer.py', 'compute_advantage',
        namespace=dict(torch=torch, np=np, Optional=Optional, DataProto=DataProto, AlgoConfig=dict,
                       core_algos=core_algos, AdvantageEstimator=core_algos.AdvantageEstimator))
    return function(batch, core_algos.AdvantageEstimator.GRAPHRPO,
                    config=OmegaConf.create({'graphrpo_alternating_roles': True}))


def _batch(role, step):
    # Three real rows: a main+branch episode and one main-only episode; one failed group.
    mask = torch.tensor([[1., 0., 1.], [1., 1., 0.], [1., 0., 1.], [0., 0., 0.]])
    if role == 'M':
        mask = torch.tensor([[0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 0., 0.]])
    reward = torch.zeros_like(mask)
    reward[:3, -1] = torch.tensor([0., 0., 1.])
    credit = mask * torch.tensor([[-1.], [-1.], [1.], [0.]])
    return DataProto.from_dict(tensors={
        'response_mask': mask, 'token_level_rewards': reward,
        'graph_decision_mask': mask.clone() if role == 'M' else torch.zeros_like(mask),
        'graph_edit_credit_mask': credit if role == 'M' else torch.full_like(mask, 9.),
        'process_reward_mask': torch.full_like(mask, -1.)},
        non_tensors={'uid': np.array(['q', 'q', 'q', 'failed'], dtype=object),
                     'gen_uid': np.array(['a', 'a', 'b', 'f'], dtype=object),
                     'graph_rpo_training_role': np.array([role] * 4, dtype=object)},
        meta_info={'global_steps': step})


def test_driver_credits_both_roles_and_shared_optimizer_updates_only_selected_tokens():
    from verl.workers.config.actor import ActorConfig

    parameter = torch.nn.Parameter(torch.zeros(4, 3))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    for role, step in [('E', 1), ('M', 2)]:
        batch = _compute(_batch(role, step))
        mask = batch.batch['response_mask']
        expected = mask * torch.tensor([[-1.], [-1.], [1.], [0.]])
        assert torch.equal(batch.batch['advantages'], expected)
        weights = batch.batch['graphrpo_loss_weights']
        assert weights[:2].sum() == 0.5 and weights[2].sum() == 0.5
        assert weights[3].sum() == 0  # A service failure is excluded, not a losing episode.
        before = parameter.detach().clone()
        loss, _ = core_algos.compute_policy_loss_graphrpo(
            old_log_prob=before, log_prob=parameter, advantages=batch.batch['advantages'],
            response_mask=mask, graphrpo_loss_weights=weights,
            config=ActorConfig(strategy='fsdp', rollout_n=1, ppo_micro_batch_size_per_gpu=1))
        optimizer.zero_grad()
        loss.backward()
        assert torch.all(parameter.grad[mask == 0] == 0)
        assert torch.all(parameter.grad[mask != 0] != 0)
        optimizer.step()
        assert torch.equal(parameter.detach()[mask == 0], before[mask == 0])
        assert not torch.equal(parameter.detach()[mask != 0], before[mask != 0])


def test_driver_rejects_wrong_role_and_executor_memory_leak():
    with pytest.raises(ValueError, match='saved training step'):
        _compute(_batch('M', 1))
    batch = _batch('E', 1)
    batch.batch['graph_decision_mask'][0, 0] = 1
    with pytest.raises(ValueError, match='exclude all M'):
        _compute(batch)


def test_executor_ties_give_zero_task_advantage():
    batch = _batch('E', 1)
    batch.batch['token_level_rewards'].zero_()
    assert not _compute(batch).batch['advantages'].any()
