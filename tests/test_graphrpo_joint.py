"""Joint-policy regressions: all sampled decisions train, penalties stay local."""
import asyncio
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from agents.graph_rpo import graph_rpo_credit_backend, reject_removed_graph_rpo_options
from tests.graph_training_fixtures import _setup, SEARCH, OPEN, finish, decision
from verl.trainer.ppo.core_algos import compute_graphrpo_advantage


@pytest.mark.parametrize('indices,penalty', [([0], 0.0), ([999], -0.3)])
def test_task_and_graph_tokens_are_jointly_trained_without_probes(monkeypatch, indices, penalty):
    graph_call = decision('prune', indices)
    responses = [SEARCH, OPEN, graph_call, finish('gold')]
    module, item, context, client, calls, closed = _setup(monkeypatch, responses)
    output, = asyncio.run(module.process_item(item, context))
    assert output.reward_score == 1
    assert len(client.calls) == len(responses)
    assert len(closed) == 1
    assert [path for path, _ in calls] == ['/search', '/open']
    sampled = ''.join(chr(t) for t, m in zip(output.response_ids, output.response_mask) if m)
    assert sampled.replace('\0', '') == ''.join(responses)
    graph_mask = output.extra_fields['graph_decision_mask']
    labels = output.extra_fields['process_reward_mask']
    assert any(graph_mask) and sum(graph_mask) < sum(output.response_mask)
    assert {v for v, m in zip(labels, graph_mask) if m} == {penalty}
    assert not any(v for v, m in zip(labels, graph_mask) if not m)
    assert not any(output.extra_fields['graph_edit_credit_mask'])


@pytest.mark.parametrize('rewards', [(0., 1.), (0., 0.)])
def test_joint_advantage_and_gradient_cover_execution_and_graph(rewards):
    # Columns: task action, graph decision, task action, padding.
    mask = torch.tensor([[1., 1., 1., 0.], [1., 1., 1., 0.]])
    terminal = torch.zeros_like(mask)
    terminal[:, 2] = torch.tensor(rewards)
    labels = torch.zeros_like(mask)
    labels[0, 1] = -.3
    advantage, _ = compute_graphrpo_advantage(
        terminal, mask, np.array(['q', 'q']), np.array(['a', 'b']),
        graph_decision_mask=torch.tensor([[0., 1., 0., 0.]] * 2),
        process_reward_mask=labels, config={'graphrpo_beta': .25},
    )
    base = 0. if rewards[0] == rewards[1] else -1.
    assert advantage[0].tolist() == pytest.approx([base, base - .075, base, 0.])
    assert advantage[1].tolist() == pytest.approx([-base, -base, -base, 0.])
    # At the old policy (ratio=1) PPO is unclipped, so every active advantage
    # contributes to the same policy update, including the graph-decision span.
    log_ratio = torch.zeros_like(mask, requires_grad=True)
    (-log_ratio.exp() * advantage * mask).sum().backward()
    assert torch.allclose(log_ratio.grad, -advantage)


@pytest.mark.parametrize('backend', ['old_policy_counterfactual_qa', 'old_policy_continuation'])
def test_removed_backends_fail(backend):
    assert graph_rpo_credit_backend({}) == 'outcome_only'
    with pytest.raises(ValueError, match='must be one of'):
        graph_rpo_credit_backend({'graph_rpo_credit_backend': backend})


@pytest.mark.parametrize('key', ['graphrpo_memory_only', 'graphrpo_alternating_roles'])
@pytest.mark.parametrize('value', [False, True])
def test_removed_role_flags_fail_even_when_false(key, value):
    with pytest.raises(ValueError, match='Removed GraphRPO options'):
        reject_removed_graph_rpo_options(OmegaConf.create({key: value}))
    with pytest.raises(ValueError, match='Removed GraphRPO options'):
        compute_graphrpo_advantage(torch.zeros(2, 1), torch.ones(2, 1),
            np.array(['q', 'q']), np.array(['a', 'b']), config={key: value})


@pytest.mark.parametrize('key', ['graph_rpo_counterfactual_samples', 'graph_rpo_continuation_samples',
                               'graph_rpo_executor_samples', 'graph_rpo_timeout_as_failure'])
def test_removed_plugin_options_fail(key):
    with pytest.raises(ValueError, match='Removed GraphRPO options'):
        reject_removed_graph_rpo_options(plugin=OmegaConf.create({key: 1}))


@pytest.mark.parametrize('setting', [
    {}, {'GRAPH_RPO_CREDIT_BACKEND': 'outcome_only'},
    {'GRAPH_RPO_CREDIT_BACKEND': 'old_policy_counterfactual_qa'},
    {'GRAPH_RPO_CREDIT_BACKEND': 'old_policy_continuation'},
    {'GRAPH_RPO_MEMORY_ONLY': '0'}, {'GRAPH_RPO_ALTERNATING_ROLES': '1'},
    {'GRAPH_RPO_COUNTERFACTUAL_SAMPLES': '2'}, {'GRAPH_RPO_CONTINUATION_SAMPLES': '2'},
])
def test_shell_preflight_rejects_removed_modes_before_gpu_setup(setting):
    bash = shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith('GRAPH_RPO_')}
    env.update(setting)
    result = subprocess.run([bash, (root / 'scripts/check_joint_graphrpo_options.sh').as_posix()],
                            env=env, capture_output=True, text=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    removed = bool(setting) and setting != {'GRAPH_RPO_CREDIT_BACKEND': 'outcome_only'}
    assert result.returncode == (2 if removed else 0), result.stderr
    if removed:
        assert 'removed GraphRPO' in result.stderr
