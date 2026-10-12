"""Model-controlled graph edits and immutable PPO prefixes, without a live model."""
import asyncio
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from agents.context_graph import ContextGraph, NodeType, NodeStatus
from agents.explicit_graph_tools import execute_graph_tool, parse_graph_call
from agents.graph_trace import GraphTraceRecorder, validate_graph_trace
from agents.session_summary import iter_agent_data
from tests.test_session_restart import make_agent
from tests.test_scienceworld_training import ACTION, Simulator, item, setup_rollout


def call(name, **args):
    return '<function=' + name + '>' + ''.join(
        f'<parameter={k}>{v}</parameter>' for k, v in args.items()) + '</function>'


def test_compaction_changes_real_input_without_losing_any_training_tokens():
    async def run():
        responses = [call('merge', node_ids='n2,n3', summary='verified result'),
                     call('graph_compact'), call('graph_read', node_id='n2'), 'finish']
        agent = make_agent(responses, enable_summary=False, graph_action_protocol='explicit_tools')
        agent.config.prompt_length = 2000
        graph = ContextGraph()
        graph.model_owned = True
        graph.add_node('fix the bug', NodeType.QUERY)
        for raw in ['obsolete evidence ' * 30, 'second evidence ' * 30]:
            graph.add_node(raw[:80], NodeType.OBSERVATION, metadata={'raw_content': raw})
        trace = GraphTraceRecorder(graph)
        agent.append({'role': 'user', 'content': 'obsolete transcript ' * 80})
        for index in range(3):
            response = await agent.step()
            archived_ids = copy.deepcopy(agent.chat_ids)
            used = agent.response_tokens_used()
            result = execute_graph_tool(graph, trace, agent, response,
                agent_name='main', turn_id=index, penalize=True)
            assert result.success
            assert agent.chat_ids == archived_ids
            if index == 1:
                assert agent.response_tokens_used() > used  # refresh is charged, never refunded
                assert len(agent.context()) < 2000
            if index == 2:
                assert 'obsolete evidence' in result  # folded raw evidence is still accessible
            agent.append({'role': 'user', 'content': str(result)})
        await agent.step()
        third_messages = agent.llm_client.calls[2][1]['messages']
        assert 'obsolete transcript' not in str(third_messages)
        assert 'obsolete evidence' not in str(third_messages)
        assert 'fix the bug' in str(third_messages) and 'verified result' in str(third_messages)
        segments = [d async for _, d in iter_agent_data({'main': agent}, True, 1)]
        assert len(segments) == 2
        learned = [t for d in segments for t, keep in zip(d['response_ids'], d['response_mask']) if keep]
        expected = [t for ts, masks in zip(agent.chat_ids, agent.token_mask) for t, keep in zip(ts, masks) if keep]
        assert learned == expected
        for data, calls in zip(segments, [agent.llm_client.calls[:2], agent.llm_client.calls[2:]]):
            sequence = data['prompt_ids'] + data['response_ids']
            assert all(sequence[:len(ids)] == ids for ids, _ in calls)
            assert all(lp == -.125 for lp, keep in zip(data['response_logprobs'], data['response_mask']) if keep)
        assert sum(sum(d['graph_decision_mask']) for d in segments) > 0
    asyncio.run(run())


@pytest.mark.parametrize('bad', [
    call('prune', node_id='n2') + ACTION,
    '<function=prune><parameter=node_id>n2</parameter>',
    '<function=prune><parameter=node_id>n2</parameter><parameter=node_id>n3</parameter></function>',
    call('add_edge', source='n1', target='n2', relation='invented'),
    call('graph_compact', ignored='x'),
])
def test_malformed_calls_rejected_before_mutation(bad):
    with pytest.raises(ValueError):
        parse_graph_call(bad)


def test_prune_recall_and_budget_rejection_preserve_raw_evidence():
    async def run():
        responses = [call('prune', node_id='n2'), call('recall', node_id='n2'), call('graph_compact')]
        agent = make_agent(responses, enable_summary=False)
        graph = ContextGraph()
        graph.model_owned = True
        graph.add_node('task', NodeType.QUERY)
        graph.add_node('brief', NodeType.OBSERVATION, metadata={'raw_content': 'x' * 2000})
        trace = GraphTraceRecorder(graph)
        for index in range(3):
            response = await agent.step()
            result = execute_graph_tool(graph, trace, agent, response, agent_name='main', turn_id=index, penalize=True)
            assert result.success == (index < 2)
            if index == 0:
                assert graph.nodes['n2'].status == NodeStatus.PRUNED
            agent.append({'role': 'user', 'content': str(result)})
        assert graph.nodes['n2'].is_active()
        assert graph.nodes['n2'].metadata['raw_content'] == 'x' * 2000
        assert not getattr(agent, 'summary_sessions', [])
        assert agent.additional_info[-2]['process_reward'] == -.3
    asyncio.run(run())


def explicit_rollout(monkeypatch, responses, sequence, max_turn=40):
    from agents.agent_text import extract_fn_call
    def structured(text):
        call = extract_fn_call(text)
        if call is None:
            return text
        args = call['arguments']
        if 'node_ids' in args:
            args['node_ids'] = args['node_ids'].split(',')
        for key in ('offset', 'topk'):
            if key in args:
                args[key] = int(args[key])
        return json.dumps({'tool': call['function'], 'arguments': args})
    module, context, client = setup_rollout(monkeypatch, [structured(r) for r in responses], max_turn=max_turn)
    class Sim(Simulator):
        pass
    Sim.sequence = sequence
    monkeypatch.setitem(sys.modules, 'scienceworld', SimpleNamespace(ScienceWorldEnv=Sim))
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.graph_action_protocol = 'explicit_tools'
    # Deliberately conflicting old settings must not trigger hidden work.
    plugin.structured_graph_controller = True
    plugin.consolidation_interval = plugin.initial_consolidation_turn = 1
    plugin.auto_prune_max_active = 2
    plugin.auto_bind_branch_edges = True
    for name in ['auto_prune_low_value', 'auto_connect_semantic', 'auto_merge_similar', 'collapse_child']:
        monkeypatch.setattr(ContextGraph, name, lambda *a, **k: pytest.fail('automatic graph maintenance'))
    return module, context, client


@pytest.mark.parametrize('teacher_profile', ['evaluation', 'graph_demonstration_v1'])
def test_model_can_decline_graph_work_and_exceed_old_active_node_limit(monkeypatch, teacher_profile):
    module, context, client = explicit_rollout(monkeypatch, [ACTION] * 14, [(0, False)] * 13 + [(100, True)])
    context.config.actor_rollout_ref.rollout.plugin.sft_teacher_profile = teacher_profile
    context.config.actor_rollout_ref.rollout.prompt_length = 30000
    outputs = asyncio.run(module.process_item(item(), context))
    trace = outputs[0].extra_fields['graph_trace']
    assert len(client.calls) == 14
    assert all(e['source'] != 'heuristic' for e in trace['events'])
    assert outputs[0].extra_fields['env_stats']['graph_n_pruned'] == 0
    assert outputs[0].extra_fields['env_stats']['consol_attempts'] == 0
    assert 'graph_compact' in str(client.calls[0][1]['messages'])
    assert ('SFT demonstration collection' in str(client.calls[0][1]['messages'])) == (teacher_profile == 'graph_demonstration_v1')
    assert outputs[0].extra_fields['env_stats']['graph_op_attempts'] == 0
    assert context.config.actor_rollout_ref.rollout.plugin.structured_graph_controller is True


def test_main_graph_calls_and_branch_graph_calls_are_policy_tokens(monkeypatch):
    responses = [call('branch', description='inspect', prompt='Inspect the room'),
                 ACTION, call('prune', node_id='n3'), call('recall', node_id='n3'),
                 call('return', message='observed room'), call('graph_read', node_id='n3'), ACTION]
    module, context, client = explicit_rollout(monkeypatch, responses, [(0, False), (100, True)])
    context.config.actor_rollout_ref.rollout.plugin.max_traj = 1  # cannot discard branch policy tokens
    outputs = asyncio.run(module.process_item(item(), context))
    assert len(client.calls) == len(responses)
    assert len(outputs) == 2 and all(o.reward_score == 1 for o in outputs)
    events = outputs[0].extra_fields['graph_trace']['events']
    edits = [e for e in events if e['op'] in ('prune', 'recall')]
    assert len(edits) == 2 and all(e['success'] and e['agent_name'] != 'main' for e in edits)
    assert 'observed' in str(client.calls[-1][1]['messages'])
    assert all(any(o.extra_fields['graph_decision_mask']) for o in outputs)
    assert outputs[0].extra_fields['env_stats']['memory_n_archives'] == 0
    assert all(not any(o.extra_fields['graph_edit_credit_mask']) for o in outputs)
    assert outputs[0].extra_fields['action_output_profile'] == 'typed_action_json_v1'
    assert outputs[0].extra_fields['graph_decision_profile'] == 'graph_affordances_v2'
    assert 'compact_token_charge=' in str(client.calls[1][1]['messages'])
    assert 'compact_token_charge=' in str(client.calls[-1][1]['messages'])
    assert all('structured_outputs' in kwargs for _, kwargs in client.calls)
    assert all('"tool"' in e['assistant_content_preview'] for e in edits)
    valid, errors, _ = validate_graph_trace(outputs[0].extra_fields['graph_trace'])
    assert valid, errors


def test_malformed_structured_action_never_reaches_simulator(monkeypatch):
    module, context, client = explicit_rollout(monkeypatch, ['{"tool":', ACTION], [(100, True)])
    outputs = asyncio.run(module.process_item(item(), context))
    stats = outputs[0].extra_fields['env_stats']
    assert stats['action_protocol_errors'] == 1
    assert stats['environment_steps'] == 1
    assert stats['completed'] == 1


@pytest.mark.parametrize('mode', ['working_context', 'legacy_cumulative'])
def test_explicit_branch_keeps_parent_final_answer_reserve(monkeypatch, mode):
    responses = [call('branch', description='inspect', prompt='Inspect the room'),
                 ACTION, call('return', message='observed room'), ACTION]
    module, context, client = explicit_rollout(monkeypatch, responses, [(0, False), (100, True)])
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.final_answer_reserve = 1024
    plugin.token_budget_mode = mode
    plugin.final_answer_safety_margin = 128
    plugin.branch_len = 1000000
    agents = []
    original = module.Agent
    class RecordingAgent(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            agents.append(self)

        async def react(self, *args, **kwargs):
            parent = agents[0]
            from agents.action_contract import BRANCH_HANDOFF_RESERVE
            remaining_before = parent.remaining_generation_tokens()
            if mode == 'working_context':
                assert self.config.response_length == parent.config.response_length
            else:
                assert self.config.response_length == remaining_before - 1152 - BRANCH_HANDOFF_RESERVE
            result = await super().react(*args, **kwargs)
            if mode == 'working_context':
                assert parent.remaining_generation_tokens() == remaining_before
            else:
                assert parent.remaining_generation_tokens() - self.response_tokens_used() >= 1152 + BRANCH_HANDOFF_RESERVE
            return result
    monkeypatch.setattr(module, 'Agent', RecordingAgent)
    outputs = asyncio.run(module.process_item(item(), context))
    assert len(agents) == 2
    assert outputs[0].extra_fields['env_stats']['consol_attempts'] == 0


def test_explicit_branch_large_observation_cannot_overrun_allowance():
    async def run():
        agent = make_agent(['search', 'return'], enable_summary=False,
                           graph_action_protocol='explicit_tools')
        async def tool(response):
            return 'large evidence ' * 10000
        await agent.react(tool, max_turn=1, max_tokens=500, summary_prompt=None)
        assert agent.response_tokens_used() <= 500
        assert len(agent.llm_client.calls) == 1
        assert agent.model_contexts[0]['response'] == 'search'
        assert agent.model_contexts[0]['output_ids']
        assert agent.model_contexts[0]['max_tokens'] <= 500
        assert len(agent.messages()[-1]['content']) < 1000
        assert agent.remaining_generation_tokens() > 5000
    asyncio.run(run())


@pytest.mark.parametrize('mode', ['working_context', 'legacy_cumulative'])
def test_exhausted_branch_report_reaches_parent_with_budget_charged(monkeypatch, mode):
    report = 'Source 42: measured room temperature; next measurement unresolved.'
    responses = [call('branch', description='inspect', prompt='Inspect room'),
                 ACTION, call('return', message=report), ACTION]
    module, context, client = explicit_rollout(monkeypatch, responses, [(0, False), (100, True)])
    rollout = context.config.actor_rollout_ref.rollout
    rollout.response_length = 20000
    rollout.plugin.token_budget_mode = mode
    rollout.plugin.branch_len = 1000000
    agents = []
    original_agent, original_action = module.Agent, module.run_action
    class RecordingAgent(original_agent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            agents.append(self)
    async def large_observation(env, response):
        result = await original_action(env, response)
        return result + ' observed evidence ' * 10000 if result else result
    monkeypatch.setattr(module, 'Agent', RecordingAgent)
    monkeypatch.setattr(module, 'run_action', large_observation)
    outputs = asyncio.run(module.process_item(item(), context))
    assert report in str(agents[0].messages())
    assert 'Branch has finished its task' not in str(agents[0].messages())
    assert agents[0].external_response_tokens == agents[1].response_tokens_used()
    if mode == 'legacy_cumulative':
        assert agents[0].response_tokens_used() <= rollout.response_length
    else:
        assert agents[0].response_tokens_used() > rollout.response_length
        assert all(len(r['input_ids']) + r['max_tokens'] <= rollout.prompt_length + rollout.response_length
                   for a in agents for r in a.model_contexts)
    assert outputs[0].extra_fields['env_stats']['forced_branch_returns'] == 1
    assert outputs[0].extra_fields['env_stats']['action_protocol_errors'] == 0


def test_more_than_ten_explicit_operations_are_not_silently_capped(monkeypatch):
    responses = [call('graph_read')] * 12 + [ACTION, ACTION]
    module, context, client = explicit_rollout(monkeypatch, responses, [(0, False), (100, True)])
    outputs = asyncio.run(module.process_item(item(), context))
    events = outputs[0].extra_fields['graph_trace']['events']
    reads = [e for e in events if e['op'] == 'graph_read']
    assert len(reads) == 12 and all(e['success'] for e in reads)
    assert len(client.calls) == len(responses)


def test_executor_compaction_exports_all_sessions_and_actual_new_prefix(monkeypatch):
    responses = [ACTION, ACTION, call('merge', node_ids='n2,n3', summary='verified current room'),
                 call('graph_compact'), ACTION]
    module, context, client = explicit_rollout(monkeypatch, responses, [(0, False), (0, False), (100, True)])
    context.config.actor_rollout_ref.rollout.prompt_length = 30000
    outputs = asyncio.run(module.process_item(item(), context))
    assert len(outputs) == 2 and all(o.reward_score == 1 for o in outputs)
    assert len(client.calls) == 5
    assert '[Model-selected active graph]' in str(client.calls[-1][1]['messages'])
    assert 'verified current room' in str(client.calls[-1][1]['messages'])
    trace = outputs[0].extra_fields['graph_trace']
    valid, errors, _ = validate_graph_trace(trace)
    assert valid, errors
    refresh = next(e for e in trace['events'] if e['op'] == 'graph_compact')
    assert refresh['context_refresh']['session_index'] == 1
    assert refresh['context_refresh']['cumulative_refresh_tokens'] > 0
    assert outputs[0].extra_fields['env_stats']['graph_context_refresh_tokens'] > 0


def test_branch_edits_cannot_receive_credit_on_a_main_turn():
    from agents.graph_rpo import valid_graph_edit_events
    event = {'source': 'model', 'success': True, 'op': 'recall', 'agent_name': '#0-inspect',
             'semantic_before_hash': 'old', 'semantic_after_hash': 'new', 'assistant_turn_index': 3}
    assert valid_graph_edit_events({'events': [event]}) == []
    event['agent_name'] = 'main'
    assert valid_graph_edit_events({'events': [event]}) == [event]


def test_graph_feedback_exposes_raw_size_and_exact_compaction_cost_without_mutating():
    from agents.explicit_graph_tools import decision_state, compaction_plan, compact_context
    agent = make_agent([], enable_summary=False, graph_action_protocol='explicit_tools')
    agent.config.prompt_length = 4000
    graph = ContextGraph(agent.tokenizer)
    graph.model_owned = True
    graph.add_node('question', NodeType.QUERY)
    graph.add_node('preview', NodeType.OBSERVATION, metadata={'raw_content': 'x' * 1000})
    before = copy.deepcopy(graph.to_dict())
    ids_before = copy.deepcopy(agent.chat_ids)
    used = agent.response_tokens_used()
    _, cost, charge = compaction_plan(agent, graph)
    state = decision_state(graph, agent)
    assert 'preview=7tok stored=1000tok' in state
    assert f'compact_prompt_tokens={cost}/4000' in state
    assert f'compact_token_charge={charge}' in state
    assert 'compact_fits_now=True' in state
    assert graph.to_dict() == before and agent.chat_ids == ids_before
    assert agent.response_tokens_used() == used
    compact_context(agent, graph)
    prefix = agent.summary_sessions[-1]['prefix']
    assert len(prefix.context()) - len(prefix.get_generation_prompt()) == cost
    assert agent.graph_context_tokens == charge
    assert agent.response_tokens_used() > used  # a real model-requested refresh is charged


def test_graph_feedback_reports_oversized_raw_graph_even_with_tiny_previews():
    from agents.explicit_graph_tools import decision_state
    agent = make_agent([], enable_summary=False)
    graph = ContextGraph(agent.tokenizer)
    graph.model_owned = True
    graph.add_node('question', NodeType.QUERY)
    graph.add_node('short', NodeType.OBSERVATION, metadata={'raw_content': 'x' * 10000})
    assert 'compact_fits_now=False' in decision_state(graph, agent)
    assert 'stored=10000tok' in decision_state(graph, agent)
    assert len(graph.active_nodes) == 2 and graph.nodes['n2'].is_active()


def test_graph_policy_examples_match_typed_schema_and_remain_optional():
    from agents.action_contract import ActionContract
    from agents.explicit_graph_tools import EXPLICIT_GRAPH_PROMPT
    contract = ActionContract('LocalSearch', None)
    examples = [line[line.index('{'):].strip() for line in EXPLICIT_GRAPH_PROMPT.splitlines()
                if '{"tool"' in line]
    assert any('"merge"' in line for line in examples)
    for example in examples:
        contract.decode(example, contract.schema())
    assert 'NOT a required workflow' in EXPLICIT_GRAPH_PROMPT
    assert 'never refunds' in EXPLICIT_GRAPH_PROMPT
