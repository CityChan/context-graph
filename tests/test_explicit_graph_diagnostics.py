from scripts.diagnose_explicit_graph_eval import task_diagnostics


def test_graph_diagnostics_separates_bookkeeping_attempts_and_successes():
    row = dict(source_index=6, status='ok', task_reward=0, is_finish=False)
    explicit = dict(source='model', decision_context={'mode': 'explicit_tool'}, op='prune')
    trajectory = dict(termination_reason='token_limit', env_stats={'consol_attempts': 0},
        graph_trace={'events': [dict(source='environment', op='observe', success=True),
            dict(explicit, success=True), dict(explicit, success=False, error='protected node')]},
        model_contexts=[{'response': 'main final response'}],
        branch_model_contexts={'branch_1': [{'response': 'branch final response'}]})
    report = task_diagnostics(row, trajectory)
    assert report['stop'] == 'token_limit'
    assert report['explicit_tool_attempts'] == {'prune': 2}
    assert report['explicit_tool_successes'] == {'prune': 1}
    assert report['last_graph_errors'][0]['error'] == 'protected node'
    assert report['events_by_source_and_op']['environment:observe'] == 1
    assert report['last_model_turns']['branch_1'][0]['response_tail'] == 'branch final response'
    assert 'passed' not in report


def test_missing_trace_and_unfinished_task_are_not_inferred_as_success():
    row = dict(source_index=28, status='ok', task_reward=0, is_finish=False)
    report = task_diagnostics(row, {})
    assert not report['trace_present']
    assert report['stop'] is None
    assert report['explicit_tool_attempts'] == {}
    assert not report['is_finish']
