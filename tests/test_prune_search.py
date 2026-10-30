import asyncio
import json
from types import SimpleNamespace

import pytest

from agents.prune_search import PruneSearchRuntime, search_key
from agents.explicit_graph_tools import configure_evaluation_protocol, execute_graph_tool
from tests.test_graph_management_full import setup
from tests.test_explicit_graph_tools import call
from tests.graph_training_fixtures import _setup


def action(tool, **arguments):
    return json.dumps(dict(tool=tool, arguments=arguments))


def rollout(monkeypatch, responses, *, train=True):
    module, item, context, client, calls, closed = _setup(monkeypatch, responses, train=train)
    configure_evaluation_protocol(context.config, 'contextgraph', 'explicit_tools')
    context.config.algorithm.update(graphrpo_invalid_action_policy='negative_all',
        graphrpo_require_binary_reward=True, graphrpo_alpha=1., graphrpo_beta=1., graphrpo_norm_adv_by_std=False)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.update(graph_rl_guard_profile='prune_search_v1', verify_training_context=True,
        graph_reject_repeated_compact=True, process_reward=['graphrpo'], graph_rpo_scope_process_reward=False)
    return module, item, context, client, calls


@pytest.mark.parametrize('reward', [0, 1])
def test_prune_success_gate_shared_cap_recall_cancellation_and_no_other_credits(reward):
    async def run():
        graph, trace, agent, _ = setup(9)
        del agent.graph_management_runtime
        runtime = PruneSearchRuntime(graph, SimpleNamespace(search_topk_cap=10), train=True)
        runtime.register(agent)
        async def edit(tool, **args):
            await agent.step()
            result = execute_graph_tool(graph, trace, agent, call(tool, **args), agent_name='main', turn_id=1, penalize=True)
            turn = len(agent.chat) - 1
            agent.append({'role': 'user', 'content': str(result)})
            return turn
        cancelled = await edit('prune', node_id='n2')
        await edit('recall', node_id='n2')
        repeated = await edit('prune', node_id='n2')
        credited = [await edit('prune', node_id=f'n{i}') for i in range(3, 10)]
        invalid = await edit('prune', node_id='n1')
        runtime.finish(reward)
        assert agent.additional_info[cancelled]['graph_edit_credit'] == 0
        assert agent.additional_info[repeated]['graph_edit_credit'] == 0
        assert sum(agent.additional_info[t]['graph_edit_credit'] for t in credited) == pytest.approx(.5 * reward)
        assert agent.additional_info[invalid]['process_reward'] == -.3
    asyncio.run(run())


@pytest.mark.parametrize('train', [True, False])
def test_real_executor_dedup_survives_prune_and_compact_without_fake_nodes(monkeypatch, train):
    responses = [action('search', query='gold'), action('prune', node_id='n2'), action('graph_compact'),
                 action('search', query=' GOLD ', topk=100), action('finish', answer='gold')]
    module, item, context, client, calls = rollout(monkeypatch, responses, train=train)
    rows = asyncio.run(module.process_item(item, context))
    assert len([c for c in calls if c[0] == '/search']) == 1
    assert rows and all(row.reward_score == 1 for row in rows)
    stats = rows[0].extra_fields['env_stats']
    assert stats['prune_search_repeat_search'] == 1
    assert stats['graph_n_nodes'] == 2
    labels = [v for row in rows for v in row.extra_fields['process_reward_mask']]
    assert set(labels) <= {0., -.3}
    assert (-.3 in labels) == train
    credits = [v for row in rows for v in row.extra_fields.get('graph_edit_credit_mask', [])]
    assert (.1 in credits) == train
    assert all(row.extra_fields['mask_rollout'] is False for row in rows)


def test_failed_search_can_retry_and_empty_success_adds_no_evidence(monkeypatch):
    from envs.local_search import AsyncSearchClient
    responses = [action('search', query='gold'), action('search', query='gold'),
                 action('search', query='gold'), action('finish', answer='gold')]
    module, item, context, _, calls = rollout(monkeypatch, responses)
    async def post(self, path, payload):
        calls.append((path, payload))
        if len(calls) == 1:
            raise RuntimeError('temporary retrieval outage')
        return []
    monkeypatch.setattr(AsyncSearchClient, '_post', post)
    rows = asyncio.run(module.process_item(item, context))
    assert len(calls) == 2
    stats = rows[0].extra_fields['env_stats']
    assert stats['prune_search_failed_searches'] == 1 and stats['prune_search_repeat_search'] == 1
    assert stats['graph_n_nodes'] == 1
    assert set(v for r in rows for v in r.extra_fields['process_reward_mask']) <= {0., -.3}


def test_format_and_graph_errors_use_point_three_without_old_unfolded_penalty(monkeypatch):
    responses = ['not JSON', action('search', query='gold'), action('prune', node_id='missing'), action('finish', answer='gold')]
    module, item, context, _, _ = rollout(monkeypatch, responses)
    rows = asyncio.run(module.process_item(item, context))
    stats = rows[0].extra_fields['env_stats']
    assert stats['prune_search_invalid_format'] == stats['prune_search_invalid_graph_action'] == 1
    assert set(v for r in rows for v in r.extra_fields['process_reward_mask']) == {0., -.3}
    assert all(r.reward_score == 1 for r in rows)


def test_effective_topk_keys_and_failed_requests_are_not_cached():
    env = SimpleNamespace(search_topk_cap=10, last_action_outcome={'success': False})
    assert search_key({'query': ' Gold '}, env) == search_key({'query': 'gold', 'topk': 100}, env)
    assert search_key({'query': 'gold', 'topk': 3}, env) != search_key({'query': 'gold'}, env)


def test_duplicate_search_shared_with_branch(monkeypatch):
    responses = [action('search', query='gold'), action('branch', description='verify', prompt='check evidence'),
                 action('search', query=' GOLD '), action('return', message='evidence retained'), action('finish', answer='gold')]
    module, item, context, _, calls = rollout(monkeypatch, responses)
    rows = asyncio.run(module.process_item(item, context))
    assert len([c for c in calls if c[0] == '/search']) == 1
    assert rows[0].extra_fields['env_stats']['prune_search_repeat_search'] == 1
    assert all(r.reward_score == 1 for r in rows)
    assert -.3 in [v for r in rows for v in r.extra_fields['process_reward_mask']]


@pytest.mark.parametrize('answer', ['gold', 'wrong'])
def test_branch_prune_credit_is_gated_by_whole_episode_outcome(monkeypatch, answer):
    responses = [action('search', query='gold'), action('branch', description='verify', prompt='check evidence'),
                 action('prune', node_id='n2'), action('return', message='evidence retained'), action('finish', answer=answer)]
    module, item, context, _, _ = rollout(monkeypatch, responses)
    rows = asyncio.run(module.process_item(item, context))
    assert all(r.reward_score == int(answer == 'gold') for r in rows)
    credits = [v for r in rows for v in r.extra_fields.get('graph_edit_credit_mask', [])]
    assert (.1 in credits) == (answer == 'gold')
    assert set(v for r in rows for v in r.extra_fields['process_reward_mask']) == {0.}


def test_forced_correct_answer_keeps_full_reward_and_prune_credit(monkeypatch):
    responses = [action('search', query='gold'), action('prune', node_id='n2'), action('finish', answer='gold')]
    module, item, context, _, _ = rollout(monkeypatch, responses)
    context.config.actor_rollout_ref.rollout.plugin.update(max_turn=2, final_answer_reserve=1024)
    rows = asyncio.run(module.process_item(item, context))
    stats = rows[0].extra_fields['env_stats']
    assert stats['forced_finish'] == 1
    assert all(r.reward_score == 1 for r in rows)
    assert .1 in [v for r in rows for v in r.extra_fields['graph_edit_credit_mask']]
    assert set(v for r in rows for v in r.extra_fields['process_reward_mask']) == {0.}


def test_new_query_without_new_documents_is_feedback_only(monkeypatch):
    responses = [action('search', query='gold'), action('search', query='other'), action('finish', answer='gold')]
    module, item, context, _, calls = rollout(monkeypatch, responses)
    rows = asyncio.run(module.process_item(item, context))
    assert len(calls) == 2 and rows[0].extra_fields['env_stats']['graph_n_nodes'] == 2
    assert set(v for r in rows for v in r.extra_fields['process_reward_mask']) == {0.}


def test_unfinished_episode_has_no_hidden_end_penalty(monkeypatch):
    module, item, context, _, _ = rollout(monkeypatch, [action('search', query='gold')])
    context.config.actor_rollout_ref.rollout.plugin.update(max_turn=1, final_answer_reserve=0)
    rows = asyncio.run(module.process_item(item, context))
    assert rows and all(r.reward_score == 0 for r in rows)
    assert rows[0].extra_fields['env_stats']['no_finish'] == 1
    assert all(r.extra_fields['mask_rollout'] is False for r in rows)
    assert set(v for r in rows for v in r.extra_fields['process_reward_mask']) == {0.}


def test_first_prune_after_recalling_folded_node_remains_eligible(monkeypatch):
    responses = [action('search', query='gold'), action('open_page', docid='1'),
        action('merge', node_ids=['n2', 'n3'], summary='evidence'), action('recall', node_id='n2'),
        action('prune', node_id='n2'), action('finish', answer='gold')]
    module, item, context, _, _ = rollout(monkeypatch, responses)
    rows = asyncio.run(module.process_item(item, context))
    assert rows[0].extra_fields['env_stats']['prune_search_credited_prunes'] == 1
    assert .1 in [v for r in rows for v in r.extra_fields['graph_edit_credit_mask']]


def test_rejected_branch_budget_gets_legality_label_without_fake_node(monkeypatch):
    responses = [action('search', query='gold'), action('branch', description='check', prompt='verify'),
        action('finish', answer='gold')]
    module, item, context, _, _ = rollout(monkeypatch, responses)
    context.config.actor_rollout_ref.rollout.plugin.max_session = 0
    rows = asyncio.run(module.process_item(item, context))
    stats = rows[0].extra_fields['env_stats']
    assert stats['prune_search_branch_budget_rejection'] == 1
    assert stats['graph_n_nodes'] == 2
    assert set(v for r in rows for v in r.extra_fields['process_reward_mask']) == {0., -.3}


def test_invalid_docid_is_rejected_without_evidence_node(monkeypatch):
    from envs.local_search import AsyncSearchClient
    module, item, context, _, _ = rollout(monkeypatch,
        [action('open_page', docid='missing'), action('search', query='gold'), action('finish', answer='gold')])
    original = AsyncSearchClient._post
    async def post(self, path, payload):
        if path == '/open':
            return [{'docid': 'missing', 'url': '', 'text': 'Document not found for given docid.'}]
        return await original(self, path, payload)
    monkeypatch.setattr(AsyncSearchClient, '_post', post)
    rows = asyncio.run(module.process_item(item, context))
    stats = rows[0].extra_fields['env_stats']
    assert stats['prune_search_invalid_environment_action'] == 1
    assert stats['graph_n_nodes'] == 2
    assert set(v for r in rows for v in r.extra_fields['process_reward_mask']) == {0., -.3}


def test_profile_rejects_automatic_maintenance_and_extra_process_rewards(monkeypatch):
    from agents.graph_rl_profile import validate_profile
    _, _, context, _, _ = rollout(monkeypatch, [])
    plugin = context.config.actor_rollout_ref.rollout.plugin
    validate_profile(context.config.algorithm, plugin)
    plugin.gm_auto_compact = True
    with pytest.raises(ValueError, match='automatic'):
        validate_profile(context.config.algorithm, plugin)
    plugin.gm_auto_compact = False
    plugin.process_reward = ['graphrpo', 'compress']
    with pytest.raises(ValueError, match='only graphrpo'):
        validate_profile(context.config.algorithm, plugin)
