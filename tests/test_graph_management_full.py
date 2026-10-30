import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.context_graph import EdgeRelation, NodeStatus
from agents.explicit_graph_tools import compaction_plan, execute_graph_tool
from agents.graph_management_reward import ledger, terminal_reward
from agents.graph_management_runtime import GraphManagementRuntime
from agents.graph_trace import GraphTraceRecorder
from tests.test_graph_management_reward import graph_with_observations
from tests.test_session_restart import make_agent
from tests.test_explicit_graph_tools import call
from verl.trainer.ppo.core_algos import compute_graphrpo_advantage


def setup(n=4, **options):
    graph = graph_with_observations(n)
    trace = GraphTraceRecorder(graph)
    agent = make_agent(['action'] * 40, enable_summary=False, token_budget_mode='working_context',
        verify_training_context=True, graph_reject_repeated_compact=True, graph_compact_node_chars=1500)
    agent.config.prompt_length = agent.prompt_length = 8192
    agent.config.response_length = agent.response_length = 57344
    runtime = GraphManagementRuntime(graph, trace, options, train=True, answers=['gold'])
    runtime.register(agent, 'main')
    return graph, trace, agent, runtime


def test_full_prompt_describes_actual_maintenance_without_changing_legacy():
    from agents.explicit_graph_tools import EXPLICIT_GRAPH_PROMPT, explicit_graph_prompt
    assert explicit_graph_prompt(SimpleNamespace()) == EXPLICIT_GRAPH_PROMPT
    plugin = SimpleNamespace(graph_rl_guard_profile='graph_management_full_v1',
                             gm_auto_prune_max_active=12, gm_auto_compact=True,
                             graph_compact_node_chars=1500)
    text = explicit_graph_prompt(plugin)
    assert 'automatic compact: enabled' in text and 'auto-prune target: 12' in text
    assert '1500 characters' in text and 'full stored content' in text
    assert 'Only this explicit call' not in text and 'automatic prune/merge' not in text
    plugin.gm_auto_prune_max_active, plugin.gm_auto_compact = 0, False
    text = explicit_graph_prompt(plugin)
    assert 'automatic compact: disabled' in text and 'auto-prune target: disabled' in text


async def sampled(agent, runtime, tool, args=None, size=100):
    await agent.step()
    runtime.sampled(agent, {'tool': tool, 'arguments': args or {}}, size)
    turn = len(agent.chat) - 1
    agent.append({'role': 'user', 'content': 'continue'})
    return turn


def reasons(agent, turn):
    return {x['reason'] for x in (agent.additional_info[turn] or {}).get('graph_management_penalties', [])}


def test_negative_all_covers_environment_tokens_ties_and_padding():
    mask = torch.tensor([[1., 1., 0.], [1., 1., 0.]])
    labels = torch.tensor([[-.2, 0., float('nan')], [0., -.1, float('nan')]])
    scores = torch.tensor([[0., 1., 0.], [0., 0., 0.]])
    args = dict(index=np.array(['q', 'q']), gen_uid=np.array(['a', 'b']), process_reward_mask=labels,
                config={'graphrpo_invalid_action_policy': 'negative_all', 'graphrpo_beta': 0.})
    advantage, _ = compute_graphrpo_advantage(scores, mask, **args)
    assert advantage[0].tolist() == pytest.approx([-.2, 1., 0.])
    assert advantage[1].tolist() == pytest.approx([-1., -1., 0.])
    advantage, _ = compute_graphrpo_advantage(scores * 0, mask, **args)
    assert advantage[0, 0] == pytest.approx(-.2)
    from tests.test_training_hardening import batch, config
    from verl.trainer.ppo.dynamic_groups import informative_groups
    data = batch((0, 0, 0))
    data.batch['process_reward_mask'][1, 0] = -.2
    assert informative_groups(data, config(graphrpo_beta=0., graphrpo_invalid_action_policy='negative_all'))[0] == [[0, 1, 2]]


def test_cumulative_edges_and_efficiency_forced_discount():
    graph = graph_with_observations()
    graph.add_edge('n2', 'n3', EdgeRelation.SEMANTIC)
    state = ledger(graph)
    state.model_edges.add(('n2', 'n3', 'semantic'))
    state.model_edge_additions = 1
    graph.nodes['n2'].status = NodeStatus.PRUNED
    base, metrics = terminal_reward(graph, 1, full=True, natural_finish=False, total_tokens=1_500_000)
    assert metrics['graph_management_density_score'] == pytest.approx(1 / 3)
    assert metrics['graph_management_model_active_edges'] == 0
    assert metrics['graph_management_discounted_task'] == .5
    assert metrics['graph_management_efficiency_bonus'] == .075
    assert base == pytest.approx(.575 + metrics['graph_management_terminal_bonus'])
    assert terminal_reward(graph, 0, full=True)[1]['graph_management_efficiency_bonus'] == 0
    assert terminal_reward(graph, 1, full=True, total_tokens=4_000_000)[1]['graph_management_efficiency_bonus'] == 0


def test_repeat_branch_pressure_rebound_and_late_answer_labels():
    async def run():
        graph, trace, agent, runtime = setup(14)
        first = await sampled(agent, runtime, 'search', {'query': 'Gold'})
        second = await sampled(agent, runtime, 'search', {'query': ' gold '})
        assert reasons(agent, first) == set()
        assert 'repeat_search' in reasons(agent, second)
        runtime.answer_seen_at = runtime.env_steps
        for i in range(10):
            last = await sampled(agent, runtime, 'open_page', {'docid': str(i)}, size=60000)
        runtime.finish(1)
        assert {'graph_neglect', 'context_pressure', 'answer_delay'} <= reasons(agent, last)
        runtime.recent_edits = [(runtime.env_steps, agent, first)]
        await sampled(agent, runtime, 'search', {'query': 'new'}, size=60000)
        assert 'pressure_rebound' in reasons(agent, first)
        for _ in range(5):
            branch = await sampled(agent, runtime, 'branch', {'prompt': 'find the same exact evidence'})
        assert {'duplicate_branch', 'excess_branch'} <= reasons(agent, branch)
        assert agent.additional_info[branch]['process_reward'] == -.3  # strongest, not sum
    asyncio.run(run())


def test_discarded_source_penalty_targets_original_edit_and_forced_return_exemption():
    async def run():
        graph, trace, agent, runtime = setup()
        graph.nodes['n2'].metadata = {'tool': 'search', 'arguments': {'query': 'evidence'}}
        await agent.step()
        result = execute_graph_tool(graph, trace, agent, call('prune', node_id='n2'), agent_name='main', turn_id=1, penalize=True)
        assert result.success
        edit = len(agent.chat) - 1
        agent.append({'role': 'user', 'content': 'done'})
        await sampled(agent, runtime, 'search', {'query': 'evidence'})
        assert 'discarded_source_retrieved' in reasons(agent, edit)
        parent_turn = await sampled(agent, runtime, 'branch', {'prompt': 'different'})
        child = make_agent(['return'], enable_summary=False)
        runtime.register(child, 'branch', parent=agent, parent_turn=parent_turn, high_pressure=False)
        child.force_action_finish = True
        child_turn = await sampled(child, runtime, 'return')
        assert 'forced_branch_return' in reasons(child, child_turn)
        assert 'forced_branch_return' in reasons(agent, parent_turn)
        exempt = make_agent(['return'], enable_summary=False)
        runtime.register(exempt, 'branch2', parent=agent, parent_turn=parent_turn, high_pressure=True)
        exempt.force_action_finish = True
        assert 'forced_branch_return' not in reasons(exempt, await sampled(exempt, runtime, 'return'))
    asyncio.run(run())


def test_truncated_compact_keeps_full_read_and_framework_segments_are_exact():
    async def run():
        graph, trace, agent, runtime = setup(gm_auto_compact=True, gm_auto_compact_max=1)
        graph.nodes['n2'].metadata['raw_content'] = 'z' * 3000
        prompt, _, _ = compaction_plan(agent, graph)
        assert 'z' * 1501 not in prompt[-1]['content']
        assert graph.nodes['n2'].metadata['raw_content'] == 'z' * 3000
        await agent.step()
        read = execute_graph_tool(graph, trace, agent, call('graph_read', node_id='n2', offset='1500'), agent_name='main', turn_id=0, penalize=True)
        assert 'z' * 1500 in str(read)
        agent.append({'role': 'user', 'content': 'obsolete ' * 6000})
        runtime.maintain(agent)
        assert runtime.auto_compacts == 1
        assert agent.summary_sessions[-1]['reason'] == 'framework_graph_compact'
        assert trace.events[-1]['source'] == 'framework'
        assert ledger(graph).credit == 0
        await agent.step()
        rows = [row async for row in agent.session_data()]
        assert sum(row['verified_sampled_turns'] for row in rows) == 2
        assert not any(v for row in rows for v in row['graph_edit_credit_mask'])
        assert 'obsolete' not in str(agent.llm_client.calls[-1][1]['messages'])
    asyncio.run(run())


def test_auto_maintenance_shared_caps_cooldown_and_gain_skip():
    graph, trace, agent, runtime = setup(15, gm_auto_prune_max_active=12, gm_auto_compact=True,
                                       gm_auto_compact_max=2, gm_auto_compact_cooldown=4)
    runtime.maintain(agent)
    assert len(graph.active_nodes) <= 12
    assert trace.events[-1]['source'] == 'framework'
    runtime.maintain(agent, pending_tokens=60000)
    assert runtime.metrics['auto_compact_gain_skips'] == 1
    agent.append({'role': 'user', 'content': 'obsolete ' * 6000})
    runtime.maintain(agent)
    assert runtime.auto_compacts == 1
    child = make_agent([], enable_summary=False, token_budget_mode='working_context')
    child.config.prompt_length = child.prompt_length = 8192
    child.config.response_length = child.response_length = 57344
    runtime.register(child, 'branch')
    child.append({'role': 'user', 'content': 'different old context ' * 2400})
    runtime.maintain(child)
    assert runtime.auto_compacts == 1
    runtime.env_steps = 4
    runtime.maintain(child)
    assert runtime.auto_compacts == 2
    assert ledger(graph).credit == 0


def test_full_rollout_connects_shaping_and_native_context_audit(monkeypatch):
    from tests.graph_training_fixtures import _setup
    from agents.explicit_graph_tools import configure_evaluation_protocol
    payloads = [dict(tool='search', arguments={'query': 'gold'})] * 2 + [dict(tool='finish', arguments={'answer': 'gold'})]
    module, item, context, _, _, _ = _setup(monkeypatch, [json.dumps(p) for p in payloads])
    configure_evaluation_protocol(context.config, 'contextgraph', 'explicit_tools')
    context.config.algorithm.update(graphrpo_invalid_action_policy='negative_all', graphrpo_require_binary_reward=False,
        graphrpo_alpha=1., graphrpo_beta=1., graphrpo_norm_adv_by_std=False)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.update(graph_rl_guard_profile='graph_management_full_v1', graph_rpo_credit_backend='management',
        verify_training_context=True, graph_reject_repeated_compact=True, graph_compact_node_chars=1500,
        gm_auto_prune_max_active=12, gm_auto_compact=True, graph_rpo_scope_process_reward=False)
    rows = asyncio.run(module.process_item(item, context))
    assert rows and all(row.extra_fields['native_task_reward'] == 1 for row in rows)
    assert 1 < rows[0].reward_score < 1.3
    assert -.2 in rows[0].extra_fields['process_reward_mask']
    assert rows[0].extra_fields['env_stats']['graph_management_penalty_repeat_search'] == 1
    assert sum(row.extra_fields['training_context_verified_turns'] for row in rows) == 3


def test_answer_delay_proxy_uses_visible_feedback_not_hidden_raw_result():
    from agents.finalizer import append_observation_preserving_final_answer
    async def run():
        _, _, agent, runtime = setup()
        await agent.step()
        agent.working_context_limit = 2000
        raw = 'x' * 3000 + ' gold ' + 'x' * 3000
        runtime.observation(agent, {'tool': 'search'}, raw)
        assert runtime.answer_seen_at is None
        visible = append_observation_preserving_final_answer(agent, raw, 512, 0)
        assert visible is not None and 'gold' not in visible
        assert runtime.answer_seen_at is None
        runtime.observation(agent, {'tool': 'search'}, 'gold')
        append_observation_preserving_final_answer(agent, 'visible gold', 0)
        assert runtime.answer_seen_at == 0
    asyncio.run(run())


def test_automatic_compact_runs_before_final_answer_headroom_check():
    from agents.finalizer import step_preserving_final_answer
    async def run():
        _, _, agent, runtime = setup(gm_auto_compact=True)
        agent.append({'role': 'user', 'content': 'old feedback ' * 5500})
        assert agent.remaining_generation_tokens() < 512
        assert await step_preserving_final_answer(agent, 512) == 'action'
        assert runtime.auto_compacts == 1
    asyncio.run(run())


@pytest.mark.parametrize('closed', [True, False])
def test_native_rl_thinking_dispatch_and_exact_policy_tokens(closed):
    from agents.action_contract import ActionContract
    from agents.utils import Agent, CallLLM
    async def run():
        base = make_agent([], enable_summary=False, explicit_action_thinking=True,
                          graph_action_protocol='explicit_tools', verify_training_context=True)
        # A tokenizer whose special thinking delimiter disappears during ordinary decoding.
        class Tokenizer(type(base.tokenizer)):
            def encode(self, text, **kwargs):
                if text == '</think>':
                    return [900000]
                return super().encode(text, **kwargs)
            def decode(self, ids, **kwargs):
                return ''.join('' if t == 900000 else chr(t) for t in ids if t != 0)
        tokenizer = Tokenizer()
        target = json.dumps({'tool': 'search', 'arguments': {'query': 'evidence'}})
        ids = tokenizer.encode('Review evidence.') + ([900000] if closed else []) + tokenizer.encode(target) + [0]
        sent = []
        async def generate(**kwargs):
            sent.append(kwargs)
            return SimpleNamespace(token_ids=ids, log_probs=[-.25] * len(ids), finish_reason='stop')
        client = CallLLM(SimpleNamespace(generate=generate), tokenizer, base.config,
                          asyncio.get_running_loop(), sampling_params={'logprobs': True})
        agent = Agent(client, base.chat, tokenizer, base.config,
                      action_contract=ActionContract('LocalSearch', None))
        wire = await agent.step()
        assert ('<function=search>' in wire) == closed
        assert agent.action_protocol_errors == int(not closed)
        assert 'structured_outputs' not in sent[0]['sampling_params']
        assert agent.chat_completions[-1]['choices'][0]['message']['raw_output_ids'] == ids
        rows = [r async for r in agent.session_data()]
        assert sum(r['verified_sampled_turns'] for r in rows) == 1
        learned = [t for r in rows for t, m in zip(r['response_ids'], r['response_mask']) if m]
        assert learned == ids
        assert all(p == -.25 for r in rows for p, m in zip(r['response_logprobs'], r['response_mask']) if m)
    asyncio.run(run())
