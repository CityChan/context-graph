"""Compression recovers request space, not lifetime accounting or model turns."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from agents import agentfold_agent as af, supo_agent as supo, memory_baseline_agent as memory
from agents.token_budget import TokenBudget
from agents.explicit_graph_tools import compact_context, decision_state
from agents.context_graph import ContextGraph, NodeType
from agents.rollout_status import classify_rollout_status
from tests.test_agentfold import setup as setup_fold, SEARCH, FINISH, compress
from tests.test_supo import setup as setup_supo, SUMMARY
from tests.test_graph_memory_baselines import setup as setup_memory, PATCH, ANALYSIS, EVOLUTION
from tests.test_session_restart import make_agent


def assert_window_run(out, context):
    stats = out.extra_fields['env_stats']
    config = context.config.actor_rollout_ref.rollout
    window = config.prompt_length + config.response_length
    records = out.extra_fields['model_contexts']
    assert out.extra_fields['is_finish']
    assert stats['cumulative_accounted_tokens'] > config.plugin.val_response_length
    assert stats['episode_token_limit'] is None
    assert stats['token_budget_mode'] == 'working_context'
    assert all(len(r['input_ids']) + r['max_tokens'] <= window for r in records)
    assert all(len(r['input_ids']) + len(r['output_ids']) <= window for r in records)
    assert not stats['finalizer_attempted']


def test_agentfold_continues_after_lifetime_limit_with_folded_history(monkeypatch):
    replies = [SEARCH] + [compress(1, i) + SEARCH for i in range(1, 9)] + [FINISH]
    env, context, item, _ = setup_fold(monkeypatch, replies, observation='evidence ' * 500)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.token_budget_mode = 'working_context'
    plugin.val_response_length = 2000  # Old cumulative cutoff, NOT the physical window.
    plugin.val_max_turn = 20
    out = asyncio.run(af.process_item(item, context))
    assert len(env.actions) == 10 and out.extra_fields['env_stats']['agentfold_folds'] == 8
    assert out.extra_fields['env_stats']['cumulative_accounted_tokens'] > 32768
    assert_window_run(out, context)


def test_supo_continues_after_summary_and_lifetime_limit(monkeypatch):
    env, context, item, _ = setup_supo(monkeypatch, [SEARCH, SUMMARY, SEARCH, SUMMARY, FINISH], observation='evidence ' * 500)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.token_budget_mode = 'working_context'
    plugin.val_response_length = 200
    out = asyncio.run(supo.process_item(item, context))
    assert len(env.actions) == 3 and out.extra_fields['env_stats']['summary_restarts'] == 2
    assert_window_run(out, context)


@pytest.mark.parametrize('method', ['amem', 'memobrain'])
def test_memory_helper_cost_never_exhausts_window_by_accumulation(monkeypatch, method):
    replies = ([SEARCH, json.dumps(PATCH), SEARCH, json.dumps(PATCH), FINISH] if method == 'memobrain'
               else [SEARCH, json.dumps(ANALYSIS), SEARCH, json.dumps(ANALYSIS), json.dumps(EVOLUTION), FINISH])
    env, context, item, _ = setup_memory(monkeypatch, method, replies)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.token_budget_mode = 'working_context'
    plugin.val_response_length = 200
    out = asyncio.run(memory.process_item(item, context))
    assert len(env.actions) == 3 and out.extra_fields['env_stats']['memory_requests'] > 0
    assert_window_run(out, context)


def test_explicit_compaction_can_recover_full_window_without_losing_policy():
    async def run():
        agent = make_agent(['inspect', 'continue'], enable_summary=False, token_budget_mode='working_context')
        agent.config.prompt_length = 1500
        agent.config.response_length = 1500
        await agent.step()
        agent.append({'role': 'user', 'content': 'old evidence ' * 210})
        graph = ContextGraph()
        graph.add_node('task', NodeType.QUERY)
        graph.add_node('verified concise evidence', NodeType.OBSERVATION)
        before = copy.deepcopy(agent.chat_ids)
        used = agent.response_tokens_used()
        assert used > agent.config.response_length
        assert 'compact_fits_now=True' in decision_state(graph, agent)
        assert compact_context(agent, graph).success
        assert agent.chat_ids == before and agent.response_tokens_used() > used
        agent.external_response_tokens = 100000  # A branch's past cost cannot fill MAIN's window.
        assert await agent.step() == 'continue'
        ids, call = agent.llm_client.calls[-1]
        assert call['max_len'] <= 3000 and len(ids) < 1500
    asyncio.run(run())


def test_oversized_actual_context_cannot_issue_request():
    agent = make_agent([], token_budget_mode='working_context')
    agent.append({'role': 'user', 'content': 'x' * 7000})
    assert asyncio.run(agent.step()) is None
    assert not agent.llm_client.calls and agent.budget_exhausted


def test_lifetime_cost_does_not_misclassify_window_stop():
    status = classify_rollout_status(response_tokens=90000, response_limit=24576,
        main_context_tokens=5000, working_context_limit=32768, is_finish=False,
        iteration=100, max_turn=100, timed_out=False, token_budget_mode='working_context')
    assert status['termination_reason'] == 'max_turn' and not status['hit_token_limit']


def test_smoke_audit_accepts_large_lifetime_cost_but_rejects_oversized_request(monkeypatch, tmp_path):
    from scripts import audit_graph_memory_smoke as audit
    record = {'input_ids': [1] * 20, 'output_ids': [2] * 10}
    stats = dict(main_len=1000, generated_tokens=10, observation_tokens=980,
                 instruction_tokens=10, memory_requests=1, memory_updates=1, memory_nodes=1)
    trajectory = dict(env_stats=stats, model_contexts=[record], baseline_protocol={'test': True})
    manifest = {'method': 'memobrain', 'baseline_protocol': {'test': True}, 'config': {'actor_rollout_ref': {'rollout': {
        'prompt_length': 40, 'response_length': 60,
        'plugin': {'val_response_length': 60, 'token_budget_mode': 'working_context'}}}}}
    (tmp_path / 'trajectory-0.json').write_text(json.dumps(trajectory))
    request_path = tmp_path / 'requests-0.jsonl'
    request_path.write_text(json.dumps(record))
    monkeypatch.setattr(audit, 'read_evaluation', lambda root: ([manifest], [{'source_index': 0}],
        dict(execution_errors=0, judge_parse_failures=0, format_retry_failures=0)))
    monkeypatch.setattr(audit, 'degeneration_stats', lambda *a: {})
    monkeypatch.setattr(audit, 'require_generation_quality', lambda *a: None)
    assert audit.audit(tmp_path)['passed']
    record['input_ids'] = [1] * 100
    request_path.write_text(json.dumps(record))
    (tmp_path / 'trajectory-0.json').write_text(json.dumps(trajectory))
    with pytest.raises(ValueError, match='exceeded working context'):
        audit.audit(tmp_path)


@pytest.mark.parametrize('method', ['react', 'foldagent', 'contextgraph', 'agentfold', 'supo', 'memobrain', 'amem'])
def test_eval_protocol_consistent_across_methods(monkeypatch, method):
    from scripts.eval_bcp_qwen38 import config_for as bcp
    from scripts.eval_agent_benchmarks import config_for as simulator
    from scripts.eval_swebench_verified import config_for as swe
    monkeypatch.delenv('AGENT_BUDGET_MODE', raising=False)
    configs = [bcp(SimpleNamespace(method=method, memory_mode='repaired')),
               swe(SimpleNamespace(method=method, context_length=65536, max_turn=100,
                                   task_timeout=3600, memory='8g', cpus=2))]
    if method != 'react':  # Simulator entry point currently exposes six methods.
        configs.append(simulator('scienceworld', method, 65536))
    for config in configs:
        budget = TokenBudget.from_config(config.actor_rollout_ref.rollout)
        assert budget.use_window and budget.metadata()['episode_token_limit'] is None
    assert configs[0].actor_rollout_ref.rollout.plugin.final_answer_reserve == 1024
    assert configs[1].actor_rollout_ref.rollout.plugin.final_answer_reserve == 0
    if len(configs) > 2:
        assert configs[2].actor_rollout_ref.rollout.plugin.final_answer_reserve == 0
    monkeypatch.setenv('AGENT_BUDGET_MODE', 'legacy_cumulative')
    assert not TokenBudget.from_config(bcp(SimpleNamespace(method=method, memory_mode='repaired')).actor_rollout_ref.rollout).use_window
