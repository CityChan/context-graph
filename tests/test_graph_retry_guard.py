import asyncio
import json
from types import SimpleNamespace

import pytest

from agents.action_contract import ActionContract, _validate
from agents.context_graph import ContextGraph, EdgeRelation, NodeType, NodeStatus
from agents.graph_controller import GraphActionController, GraphControllerError
from agents.graph_retry_guard import GraphRetryGuard, configured_guard


def setup_graph():
    g = ContextGraph()
    g.add_node('task', NodeType.QUERY)
    g.add_node('first evidence', NodeType.OBSERVATION)
    g.add_node('second evidence', NodeType.OBSERVATION)
    g.model_owned = True
    guard = GraphRetryGuard(g)
    g.retry_guard = guard
    return g, guard


def test_failure_cache_ignores_accounting_but_reopens_when_target_changes():
    graph, guard = setup_graph()
    call = {'function': 'recall', 'arguments': {'node_id': 'n2'}}
    guard.record(call, False)
    graph.record_graph_op(False)
    graph.add_node('unrelated evidence', NodeType.OBSERVATION)
    assert guard.blocked(call)
    graph.nodes['n2'].status = NodeStatus.FOLDED
    assert guard.blocked(call) is None


def test_three_failures_cannot_be_reset_by_compaction_read_or_pass():
    graph, guard = setup_graph()
    call = {'function': 'prune', 'arguments': {'node_id': 'missing'}}
    for _ in range(3):
        guard.record(call, False, input_tokens=200, output_tokens=30)
    assert guard.task_only
    for name in ('graph_compact', 'graph_read', 'pass'):
        guard.record({'function': name, 'arguments': {}}, True)
        assert guard.task_only
    contract = ActionContract('ScienceWorld@real', None, retry_guard=guard)
    tools = [s['properties']['tool']['const'] for s in contract.schema()['anyOf']]
    assert tools == ['branch', 'action']
    guard.task_executed()
    assert not guard.task_only
    assert guard.blocked(call)  # task execution does not erase failed-target cache
    assert guard.stats()['failed_input_tokens'] == 600
    assert guard.stats()['failed_output_tokens'] == 90
    assert guard.stats()['cooldowns'] == 1


def test_main_and_branch_share_cooldown_and_preserve_return():
    graph, guard = setup_graph()
    main = ActionContract('ScienceWorld@real', None, retry_guard=guard)
    branch = ActionContract('ScienceWorld@real', None, branch=True, retry_guard=guard)
    for _ in range(3):
        guard.record({'function': 'merge', 'arguments': {'node_ids': 'missing'}}, False)
    for contract in (main, branch):
        assert 'merge' not in str(contract.schema())
    _validate({'tool': 'return', 'arguments': {'message': 'report'}}, branch.schema())


def test_opt_in_only_and_training_rejected():
    graph, _ = setup_graph()
    assert configured_guard(graph, SimpleNamespace(), is_train=False) is None
    with pytest.raises(ValueError, match='inference'):
        configured_guard(graph, SimpleNamespace(graph_retry_guard=True), is_train=True)
    with pytest.raises(ValueError, match='positive'):
        GraphRetryGuard(graph, 0)


def test_dynamic_explicit_targets_and_edge_legality():
    graph, guard = setup_graph()
    graph.select('n2')
    graph.add_edge('n2', 'n3', EdgeRelation.SEMANTIC)
    schema = ActionContract('ScienceWorld@real', None, retry_guard=guard).schema()
    bad = [
        {'tool': 'merge', 'arguments': {'node_ids': ['n1', 'n2'], 'summary': 's'}},
        {'tool': 'prune', 'arguments': {'node_id': 'n1'}},
        {'tool': 'select', 'arguments': {'node_id': 'n2'}},
        {'tool': 'recall', 'arguments': {'node_id': 'n3'}},
        {'tool': 'add_edge', 'arguments': {'source': 'n2', 'target': 'n2', 'relation': 'causal'}},
        {'tool': 'add_edge', 'arguments': {'source': 'n2', 'target': 'n3', 'relation': 'semantic'}},
    ]
    for action in bad:
        with pytest.raises(ValueError):
            _validate(action, schema)
    _validate({'tool': 'merge', 'arguments': {'node_ids': ['n2', 'n3'], 'summary': 'evidence'}}, schema)
    _validate({'tool': 'add_edge', 'arguments': {'source': 'n3', 'target': 'n2', 'relation': 'causal'}}, schema)


def test_guarded_controller_enforces_arity_and_unused_fields():
    graph, guard = setup_graph()
    controller = GraphActionController(retry_guard=guard)
    snapshot = controller.snapshot(graph)
    schema = controller.structured_outputs(snapshot, allow_pass=True)['json']
    good = {'action': 'merge', 'candidate_indices': [0, 1], 'summary': 'facts', 'relation': 'semantic'}
    _validate(good, schema)
    assert controller.resolve_action(graph, snapshot, json.dumps(good), allow_pass=True)['function'] == 'merge'
    for bad in [dict(good, candidate_indices=[0]), dict(good, action='prune'),
                dict(good, action='pass'), dict(good, summary='')]:
        with pytest.raises(GraphControllerError):
            controller.resolve_action(graph, snapshot, json.dumps(bad), allow_pass=True)


def test_guarded_controller_excludes_duplicate_edges_and_current_focus():
    graph, guard = setup_graph()
    graph.select('n2')
    graph.add_edge('n2', 'n3', EdgeRelation.SEMANTIC)
    c = GraphActionController(retry_guard=guard)
    s = c.structured_outputs(c.snapshot(graph), allow_pass=True)['json']
    for name, ids in [('select', [0]), ('add_edge', [0, 0]), ('add_edge', [0, 1])]:
        with pytest.raises(ValueError):
            _validate({'action': name, 'candidate_indices': ids, 'summary': '', 'relation': 'semantic'}, s)
    _validate({'action': 'pass', 'candidate_indices': [], 'summary': '', 'relation': 'semantic'}, s)


def test_real_compaction_does_not_reset_failure_memory():
    from agents.explicit_graph_tools import execute_graph_tool
    from agents.graph_trace import GraphTraceRecorder
    from tests.test_session_restart import make_agent
    from tests.test_explicit_graph_tools import call

    async def run():
        graph, guard = setup_graph()
        agent = make_agent([call('prune', node_id='missing'), call('graph_compact'),
                            call('prune', node_id='missing')], enable_summary=False,
                           graph_action_protocol='explicit_tools', token_budget_mode='working_context')
        agent.config.prompt_length = 4000
        agent.append({'role': 'user', 'content': 'old transcript ' * 600})
        trace = GraphTraceRecorder(graph)
        for i, expected in enumerate([False, True, False]):
            response = await agent.step()
            result = execute_graph_tool(graph, trace, agent, response, agent_name='main', turn_id=i, penalize=False)
            assert result.success == expected
            agent.append({'role': 'user', 'content': str(result)})
        assert len(agent.summary_sessions) == 1
        assert guard.consecutive_failures == 2
        assert guard.counters['blocked_attempts'] == 1
        assert graph.invalid_op_count == 2
    asyncio.run(run())


def test_evaluator_64k_applies_to_validation_and_guard_is_opt_in(monkeypatch):
    from scripts.eval_bcp_qwen38 import config_for
    args = SimpleNamespace(method='contextgraph', memory_mode='repaired', graph_action_protocol='controller')
    monkeypatch.setenv('EVAL_CONTEXT_LENGTH', '65536')
    monkeypatch.setenv('GRAPH_RETRY_GUARD', '1')
    config = config_for(args).actor_rollout_ref.rollout
    assert config.prompt_length + config.response_length == 65536
    assert config.plugin.val_response_length == 57344
    assert config.plugin.graph_retry_guard is True
    monkeypatch.delenv('GRAPH_RETRY_GUARD')
    assert config_for(args).actor_rollout_ref.rollout.plugin.graph_retry_guard is False


def test_controller_failures_reach_attempt_budget_in_real_loop(monkeypatch):
    from tests.test_scienceworld_training import ACTION, Simulator, item, setup_rollout
    import sys
    class Sim(Simulator):
        sequence = [(0, False)] * 8 + [(100, True)]
    responses = [ACTION, '{"action":"merge","candidate_indices":[0],"summary":"","relation":"semantic"}',
                 ACTION, '{"action":"prune","candidate_indices":[],"summary":"","relation":"semantic"}', ACTION]
    module, context, client = setup_rollout(monkeypatch, responses, train=False, max_turn=5)
    monkeypatch.setitem(sys.modules, 'scienceworld', SimpleNamespace(ScienceWorldEnv=Sim))
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.graph_retry_guard = True
    plugin.structured_graph_controller = True
    plugin.graph_controller_counts_as_turn = True
    plugin.consolidation_interval = 1
    plugin.auto_prune_max_active = 0
    plugin.capture_model_contexts = True
    plugin.scienceworld_training_reward_profile = 'binary'
    outputs = asyncio.run(module.process_item(item(), context))
    stats = outputs[0].extra_fields['env_stats']
    assert stats['consol_controller_errors'] == 2
    assert stats['graph_invalid_ops'] == stats['graph_op_attempts'] == 2
    assert stats['graph_retry_guard']['failures'] == 2
    assert stats['graph_retry_guard']['failed_output_tokens'] > 0
    assert len(client.calls) == 5


def test_guarded_explicit_real_loop_recovers_from_three_bad_decisions(monkeypatch):
    from tests.test_explicit_graph_tools import explicit_rollout, call
    from tests.test_scienceworld_training import ACTION, item
    responses = [call('prune', node_id='missing')] * 3 + [ACTION]
    module, context, client = explicit_rollout(monkeypatch, responses, [(100, True)], max_turn=8)
    context.is_train = False
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.graph_retry_guard = True
    plugin.scienceworld_training_reward_profile = 'binary'
    outputs = asyncio.run(module.process_item(item(), context))
    stats = outputs[0].extra_fields['env_stats']
    assert stats['graph_retry_guard']['cooldowns'] == 1
    assert stats['graph_invalid_ops'] == 3
    assert stats['environment_steps'] == 1
    last_schema = client.calls[-1][1]['structured_outputs']['json']
    assert all(c['properties']['tool']['const'] not in {'prune', 'merge', 'graph_compact'} for c in last_schema['anyOf'])


def test_paired_audit_rejects_changed_protocol_and_incomplete_outputs(tmp_path):
    from scripts.audit_graph_retry_ab import audit
    for variant, enabled in [('baseline', False), ('guarded', True)]:
        arm = tmp_path / variant
        arm.mkdir()
        for rank in range(3):
            manifest = {'rank': rank, 'indices': [7], 'seed': 42, 'model': 'same',
                        'config': {'actor_rollout_ref': {'rollout': {'prompt_length': 8192,
                            'response_length': 57344, 'plugin': {'val_response_length': 57344,
                            'graph_retry_guard': enabled}}}}}
            (arm / f'manifest-{rank}.json').write_text(json.dumps(manifest))
            row = {'source_index': 7, 'task_id': 'q', 'status': 'ok', 'task_reward': 0,
                   'env_stats': {'token_budget_mode': 'working_context',
                                 'graph_retry_guard': {'enabled': enabled, 'requests': 0}}}
            (arm / f'results-{rank}.jsonl').write_text(json.dumps(row) + '\n' if rank == 0 else '')
        (arm / 'trajectory-7.json').write_text('{}')
        (arm / 'summary.json').write_text('{"generation_quality_passed":true}')
    result = audit(tmp_path)
    assert result['passed'] and not result['guard_exercised']
    path = tmp_path / 'guarded/manifest-0.json'
    manifest = json.loads(path.read_text())
    manifest['seed'] = 43
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='differs'):
        audit(tmp_path)
    manifest['seed'] = 42
    path.write_text(json.dumps(manifest))
    (tmp_path / 'guarded/results-0.jsonl').write_text('')
    with pytest.raises(AssertionError, match='incomplete'):
        audit(tmp_path)


def test_generated_guard_schemas_are_valid_json_schema():
    import jsonschema
    from scripts.preflight_graph_retry_schema import schemas
    for _, schema in schemas():
        jsonschema.Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize('paired', [False, True])
def test_shared_server_evaluation_loop_passes_flag_to_all_shards(tmp_path, paired):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash:
        pytest.skip('Bash required')
    root = Path(__file__).resolve().parents[1]
    source = (root / 'scripts/eval_bcp_qwen38_4node_idev.sh').read_text()
    loop = source[source.index('SUITE_ROOT=$RUN_ROOT'):]
    script = tmp_path / 'loop.sh'
    script.write_text('set -euo pipefail\nMETHOD=contextgraph\nBENCHMARK=bcp\n'
                      'NODES=(a b c d)\nMODEL_IPS=(b c d)\nMODEL_PORT=18000\nSELF=stub\n'
                      'srun() { printf "%s %s %s\\n" "$RUN_ROOT" "$GRAPH_RETRY_GUARD" "${@: -1}" > "$RUN_ROOT/captured-${@: -1}"; }\n'
                      'python() { printf "%s\\n" "$*" > "$RUN_ROOT/merge-called"; }\n' + loop)
    run = tmp_path / 'fresh-run'
    run.mkdir()
    env = dict(os.environ, RUN_ROOT=run.as_posix(), GRAPH_RETRY_AB=str(int(paired)), GRAPH_RETRY_GUARD='0')
    result = subprocess.run([bash, script.as_posix()], env=env, capture_output=True, text=True,
                            timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stderr
    for variant, flag in ([('baseline', 0), ('guarded', 1)] if paired else [('', 0)]):
        directory = run / variant
        for rank in range(3):
            assert (directory / f'captured-{rank}').read_text().strip().endswith(f' {flag} {rank}')
        assert (directory / 'merge-called').is_file()
