import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from agents.context_graph import ContextGraph, NodeType, NodeStatus, EdgeRelation
from agents.graph_management_reward import ledger, record_operation, terminal_reward
from agents.graph_rl_profile import validate_profile


def graph_with_observations(n=4):
    graph = ContextGraph()
    graph.add_node('task', NodeType.QUERY)
    for i in range(n):
        graph.add_node(str(i), NodeType.OBSERVATION, parent_id=graph.root_id)
    return graph


def event(before, after, op, success=True, source='model'):
    return dict(before_state=before, after_state=after, op=op, success=success, source=source)


def test_terminal_components_model_edges_only_and_small_graph_gate():
    graph = graph_with_observations()
    assert terminal_reward(graph, 0)[0] == pytest.approx(.08)  # size term, even on a wrong answer
    # Framework edges do not earn the model-edge density score.
    assert terminal_reward(graph, 1)[1]['graph_management_model_active_edges'] == 0
    graph.nodes['n2'].status = NodeStatus.PRUNED
    graph.add_edge('n3', 'n4', EdgeRelation.SEMANTIC)
    ledger(graph).model_edges.add(('n3', 'n4', 'semantic'))
    reward, metrics = terminal_reward(graph, 1)
    assert metrics['graph_management_archive_ratio'] == .25
    assert metrics['graph_management_density_score'] == pytest.approx(1 / 3)
    assert reward == pytest.approx(1 + .2 * (.4 * .25 + .4 + .2 / 3))
    graph.nodes['n4'].status = NodeStatus.PRUNED
    assert terminal_reward(graph, 1)[1]['graph_management_model_active_edges'] == 0
    assert terminal_reward(graph_with_observations(3), 1)[0] == 1
    large = graph_with_observations(23)
    assert terminal_reward(large, 0)[1]['graph_management_size_score'] == .5
    with pytest.raises(ValueError, match='binary'):
        terminal_reward(graph, .5)


def test_credit_only_for_effective_model_operations_and_episode_cap():
    graph = graph_with_observations(14)
    agent = SimpleNamespace(config=SimpleNamespace(prompt_length=8192, response_length=57344))
    before = graph.to_dict()
    after = copy.deepcopy(before)
    after['nodes'][1]['status'] = 'pruned'
    assert record_operation(graph, agent, event(before, before, 'prune'), 100, 100) == 0
    assert record_operation(graph, agent, event(before, after, 'prune', source='framework'), 100, 100) == 0
    assert record_operation(graph, agent, event(before, after, 'prune', success=False), 100, 100) == 0
    for _ in range(15):
        record_operation(graph, agent, event(before, after, 'prune'), 100, 100)
    assert ledger(graph).credit == pytest.approx(1)
    assert ledger(graph).credited_operations['prune'] == 10
    # Another agent on the shared graph cannot reset the per-episode cap.
    assert record_operation(graph, copy.copy(agent), event(before, after, 'merge'), 100, 100) == 0


def test_compact_credit_pressure_savings_and_two_bonus_limit():
    graph = graph_with_observations()
    agent = SimpleNamespace(config=SimpleNamespace(prompt_length=8192, response_length=57344))
    snapshot = graph.to_dict()
    compact = event(snapshot, snapshot, 'graph_compact')
    assert record_operation(graph, agent, compact, 50000, 1000) == 0
    assert record_operation(graph, agent, compact, 3000, 1001) == 0
    assert record_operation(graph, agent, compact, 3000, 1000) == .2
    assert record_operation(graph, agent, compact, 4000, 1000) == .2
    assert record_operation(graph, agent, compact, 5000, 1000) == 0
    assert ledger(graph).compact_bonuses == 2
    assert ledger(graph).effective_operations['graph_compact'] == 3


@pytest.mark.parametrize('train', [True, False])
def test_rollout_reward_and_token_credit_are_connected_with_native_outcome_retained(monkeypatch, train):
    from tests.graph_training_fixtures import _setup
    from agents.explicit_graph_tools import configure_evaluation_protocol
    payloads = [dict(tool='search', arguments={'query': str(i)}) for i in range(4)]
    payloads += [dict(tool='merge', arguments={'node_ids': ['n2', 'n3'], 'summary': 'combined evidence'}),
                 dict(tool='prune', arguments={'node_id': 'n4'}),
                 dict(tool='add_edge', arguments={'source': 'n6', 'target': 'n5', 'relation': 'semantic'}),
                 dict(tool='graph_compact', arguments={}),
                 dict(tool='finish', arguments={'answer': 'gold'})]
    module, item, context, _, _, _ = _setup(monkeypatch, [json.dumps(p) for p in payloads], train=train)
    configure_evaluation_protocol(context.config, 'contextgraph', 'explicit_tools')
    context.config.algorithm.update(graphrpo_invalid_action_policy='negative_invalid_graph',
        graphrpo_require_binary_reward=False, graphrpo_alpha=1., graphrpo_beta=1., graphrpo_norm_adv_by_std=False)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.update(graph_rl_guard_profile='graph_management_v1', graph_rpo_credit_backend='management',
        verify_training_context=True, graph_reject_repeated_compact=True, graph_rpo_scope_process_reward=False,
        max_turn=16, val_max_turn=16)
    validate_profile(context.config.algorithm, plugin)
    rows = asyncio.run(module.process_item(item, context))
    assert rows and all(row.extra_fields['native_task_reward'] == 1. for row in rows)
    if train:
        assert all(row.reward_score == pytest.approx(1.16) for row in rows)
        credits = [v for row in rows for v in row.extra_fields['graph_edit_credit_mask']]
        assert .1 in credits and .05 in credits
        stats = rows[0].extra_fields['env_stats']
        assert stats['graph_management_credited_merge'] == stats['graph_management_credited_prune'] == 1
        assert stats['graph_management_local_credit'] >= .25
        assert sum(row.extra_fields['training_context_verified_turns'] for row in rows) == 9
    else:
        assert all(row.reward_score == 1. for row in rows)


def test_management_profile_fails_closed_on_wrong_reward_normalization():
    plugin = dict(graph_rl_guard_profile='graph_management_v1', graph_rpo_credit_backend='management',
        graph_action_protocol='explicit_tools', verify_training_context=True, graph_reject_repeated_compact=True)
    algo = dict(adv_estimator='graphrpo', graphrpo_invalid_action_policy='negative_invalid_graph',
        graphrpo_require_binary_reward=False, graphrpo_alpha=1., graphrpo_beta=1., graphrpo_norm_adv_by_std=False)
    validate_profile(algo, plugin)
    for key, bad in [('graphrpo_require_binary_reward', True), ('graphrpo_alpha', .1),
                     ('graphrpo_norm_adv_by_std', True)]:
        with pytest.raises(ValueError):
            validate_profile({**algo, key: bad}, plugin)


def test_shaped_outcomes_and_operation_credit_do_not_rescue_invalid_actions():
    import numpy as np
    import torch
    from verl.trainer.ppo.core_algos import compute_graphrpo_advantage
    mask = torch.ones(2, 3)
    # Correct/incorrect task, both with structural terminal shaping.
    rewards = torch.tensor([[0., 0., 1.16], [0., 0., .08]])
    credit = torch.tensor([[.1, 0., 0.], [0., 0., 0.]])
    invalid = torch.tensor([[0., -.3, 0.], [0., 0., 0.]])
    adv, _ = compute_graphrpo_advantage(rewards, mask, np.array(['q', 'q']), np.array(['a', 'b']),
        graph_edit_credit_mask=credit, graph_decision_mask=torch.tensor([[1., 1., 0.], [0., 0., 0.]]),
        process_reward_mask=invalid, graph_invalid_penalty_mask=invalid,
        config=dict(graphrpo_require_binary_reward=False, graphrpo_alpha=1., graphrpo_beta=1.,
                    graphrpo_norm_adv_by_std=False, graphrpo_invalid_action_policy='negative_invalid_graph'))
    assert adv[0].tolist() == pytest.approx([.64, -.3, .54])
    assert adv[1].tolist() == pytest.approx([-.54, -.54, -.54])
