import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.context_graph import ContextGraph, NodeType
from agents.explicit_graph_tools import execute_graph_tool
from agents.graph_trace import GraphTraceRecorder
from agents.session_summary import iter_agent_data
from agents.training_context_audit import validate_segment
from tests.test_explicit_graph_tools import call
from tests.test_session_restart import make_agent
from verl.trainer.ppo.core_algos import compute_graphrpo_advantage


def advantage_args():
    mask = torch.tensor([[1., 1., 1., 0.]] * 3)
    rewards = torch.zeros_like(mask)
    rewards[:2, 2] = 1
    penalties = torch.tensor([[0., -.3, 0., 0.], [0., 0., 0., 0.], [0., -.3, 0., 0.]])
    # Column 0: heuristic scope penalty; column 1: verified illegal graph operation.
    process = penalties.clone()
    process[:, 0] = -.2
    return dict(token_level_rewards=rewards, response_mask=mask, index=np.array(['q'] * 3),
                gen_uid=np.array(['win', 'win', 'lose']), process_reward_mask=process,
                graph_decision_mask=torch.tensor([[0., 1., 0., 0.]] * 3), graph_invalid_penalty_mask=penalties,
                config={'graphrpo_beta': .25, 'graphrpo_invalid_action_policy': 'negative_invalid_graph'})


def test_only_verified_illegal_graph_tokens_get_negative_cap_and_gradient():
    args = advantage_args()
    adv, _ = compute_graphrpo_advantage(**args)
    assert adv[0].tolist() == pytest.approx([.95, -.3, 1., 0.])
    assert adv[1].tolist() == pytest.approx([.95, 1., 1., 0.])
    assert adv[2].tolist() == pytest.approx([-1.05, -1.075, -1., 0.])
    log_ratio = torch.zeros_like(adv, requires_grad=True)
    (-log_ratio.exp() * adv).sum().backward()
    assert log_ratio.grad[0, 1] > 0 and log_ratio.grad[0, 2] < 0
    args['config']['graphrpo_invalid_action_policy'] = 'additive'
    old, _ = compute_graphrpo_advantage(**args)
    assert old[0, 1] == pytest.approx(.925)


def test_ties_padding_exclusion_and_missing_provenance():
    args = advantage_args()
    args['token_level_rewards'].zero_()
    args['graph_invalid_penalty_mask'][:, 3] = float('nan')
    args['excluded_gen_uids'] = {'win'}
    # Give the remaining group two candidates after exclusion.
    args['gen_uid'] = np.array(['a', 'b', 'win'])
    adv, _ = compute_graphrpo_advantage(**args)
    assert adv[0, 1] == pytest.approx(-.3)
    assert not adv[2].any() and not adv[:, 3].any()
    args['graph_invalid_penalty_mask'] = None
    with pytest.raises(ValueError, match='requires graph_invalid_penalty_mask'):
        compute_graphrpo_advantage(**args)


@pytest.mark.parametrize('value', [float('nan'), .2, -2.])
def test_invalid_penalty_values_fail(value):
    args = advantage_args()
    args['graph_invalid_penalty_mask'][0, 1] = value
    with pytest.raises(ValueError, match='within'):
        compute_graphrpo_advantage(**args)


def make_compacted_agent():
    async def run():
        agent = make_agent([call('prune', node_id='missing'), call('graph_compact'), 'next task action'],
                           enable_summary=False, verify_training_context=True, graph_reject_repeated_compact=True)
        agent.config.prompt_length = agent.prompt_length = 2000
        graph = ContextGraph()
        graph.model_owned = True
        graph.add_node('fix the bug', NodeType.QUERY)
        graph.add_node('verified evidence', NodeType.OBSERVATION)
        trace = GraphTraceRecorder(graph)
        for index in range(2):
            response = await agent.step()
            result = execute_graph_tool(graph, trace, agent, response, agent_name='main', turn_id=index, penalize=True)
            assert result.success == bool(index)
            agent.append({'role': 'user', 'content': str(result)})
        await agent.step()
        return agent
    return asyncio.run(run())


def test_compaction_preserves_sampled_context_logprobs_and_failure_credit():
    agent = make_compacted_agent()
    async def export():
        return [d async for _, d in iter_agent_data({'main': agent}, True)]
    segments = asyncio.run(export())
    assert len(segments) == 2 and sum(d['verified_sampled_turns'] for d in segments) == 3
    assert min(segments[0]['graph_invalid_penalty_mask']) == -.3
    assert not any(segments[1]['graph_invalid_penalty_mask'])
    assert [d['verified_sampled_turns'] for d in segments] == [2, 1]
    for data in segments:
        assert all(lp == -.125 for lp, m in zip(data['response_logprobs'], data['response_mask']) if m)


@pytest.mark.parametrize('field', ['prompt_ids', 'response_ids', 'response_logprobs', 'process_reward_mask', 'graph_invalid_penalty_mask', 'response_mask'])
def test_context_audit_rejects_corrupted_fields(field):
    agent = make_compacted_agent()
    async def export():
        return [d async for d in agent.session_data()]
    data = copy.deepcopy(asyncio.run(export())[0])
    index = 0 if field == 'prompt_ids' else next(i for i, m in enumerate(data['response_mask']) if m)
    data[field][index] += 1
    with pytest.raises(ValueError, match='RL'):
        validate_segment(agent, 0, agent.summary_sessions[0]['start'], data)


def test_context_audit_refuses_response_truncation():
    agent = make_compacted_agent()
    async def export():
        return [d async for d in agent.session_data()]
    data = asyncio.run(export())[0]
    data['response_turn_token_indices'] = {}
    with pytest.raises(ValueError, match='RL export'):
        validate_segment(agent, 0, agent.summary_sessions[0]['start'], data)


def test_oversized_prefix_packs_native_requests_without_history_supervision():
    from agents.training_context_audit import pack_verified_segment
    agent = make_compacted_agent()
    async def export():
        return [d async for d in agent.session_data()]
    original = asyncio.run(export())
    agent.config.prompt_length = 8
    exported = asyncio.run(export())
    assert sum(row['verified_sampled_turns'] for row in exported) == 3
    assert all(len(row['prompt_ids']) == 8 for row in exported)
    for segment in original:
        rows = list(pack_verified_segment(agent, segment))
        assert len(rows) == len(segment['response_turn_token_indices'])
        for row in rows:
            turn, indices = next(iter(row['response_turn_token_indices'].items()))
            assert len(row['prompt_ids']) == 8 and len(row['response_ids']) <= agent.config.response_length
            assert not any(row['response_mask'][:indices[0]])
            assert validate_segment(agent, turn, turn + 1, row) == 1
    agent.config.response_length = 1
    with pytest.raises(ValueError, match='capacity'):
        list(pack_verified_segment(agent, original[0]))


def test_compact_duplicate_guard_allows_changed_graph_and_is_agent_local():
    async def run():
        agent = make_agent([call('graph_compact')] * 3, enable_summary=False, graph_reject_repeated_compact=True)
        agent.config.prompt_length = agent.prompt_length = 2000
        graph = ContextGraph()
        graph.model_owned = True
        graph.add_node('task', NodeType.QUERY)
        trace = GraphTraceRecorder(graph)
        for i, expected in enumerate([True, False, True]):
            if i == 2:
                graph.add_node('new evidence', NodeType.OBSERVATION)
            response = await agent.step()
            sessions = len(getattr(agent, 'summary_sessions', []))
            result = execute_graph_tool(graph, trace, agent, response, agent_name='main', turn_id=i, penalize=True)
            assert result.success == expected
            assert len(agent.summary_sessions) == sessions + int(expected)
            if not expected:
                assert agent.additional_info[-1]['graph_invalid_penalty'] == -.3
            agent.append({'role': 'user', 'content': str(result)})
        branch = make_agent([call('graph_compact')], enable_summary=False, graph_reject_repeated_compact=True)
        branch.config.prompt_length = branch.prompt_length = 2000
        assert execute_graph_tool(graph, trace, branch, await branch.step(), agent_name='branch', turn_id=0, penalize=True).success
    asyncio.run(run())


def test_native_explicit_rollout_exports_verified_invalid_decisions(monkeypatch):
    from tests.graph_training_fixtures import _setup
    from agents.explicit_graph_tools import configure_evaluation_protocol
    payloads = [dict(tool='search', arguments={'query': 'gold'}), dict(tool='prune', arguments={'node_id': 'missing'}),
                dict(tool='graph_compact', arguments={}), dict(tool='finish', arguments={'answer': 'gold'})]
    module, item, context, _, _, _ = _setup(monkeypatch, [json.dumps(p) for p in payloads])
    configure_evaluation_protocol(context.config, 'contextgraph', 'explicit_tools')
    context.config.algorithm.graphrpo_invalid_action_policy = 'negative_invalid_graph'
    context.config.actor_rollout_ref.rollout.plugin.update(graph_rl_guard_profile='invalid_action_v1',
        verify_training_context=True, graph_reject_repeated_compact=True, graph_rpo_scope_process_reward=False)
    result = asyncio.run(module.process_item(item, context))
    assert len(result) == 2  # before and after model-requested compact
    assert sum(r.extra_fields['training_context_verified_turns'] for r in result) == 4
    assert min(result[0].extra_fields['graph_invalid_penalty_mask']) == -.3
    assert not any(result[1].extra_fields['graph_invalid_penalty_mask'])


def test_launcher_syntax():
    import subprocess
    for name in ['train_bcp_qwen35_9b_guarded_rl.sh', 'train_bcp_qwen35_9b_50step.sh', 'train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh']:
        subprocess.run(['bash', '-n', str(Path('scripts') / name)], check=True)


@pytest.mark.parametrize('setting', ['auto_prune_max_active', 'enable_retrieval_memory',
    'consolidation_interval', 'auto_bind_branch_edges', 'graph_rpo_scope_process_reward'])
def test_profile_rejects_automatic_maintenance_or_scope_shaping(setting):
    from agents.graph_rl_profile import validate_profile
    algorithm = dict(adv_estimator='graphrpo', graphrpo_invalid_action_policy='negative_invalid_graph')
    plugin = dict(graph_rl_guard_profile='invalid_action_v1', graph_action_protocol='explicit_tools',
        graph_rpo_credit_backend='outcome_only', verify_training_context=True, graph_reject_repeated_compact=True)
    validate_profile(algorithm, plugin)
    plugin[setting] = True
    with pytest.raises(ValueError, match='forbids'):
        validate_profile(algorithm, plugin)
