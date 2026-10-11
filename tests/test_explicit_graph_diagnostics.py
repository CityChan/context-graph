from scripts.diagnose_explicit_graph_eval import task_diagnostics, model_turn


def test_old_contextgraph_input_records_do_not_fabricate_empty_outputs():
    report = model_turn({'input_ids': [1, 2, 3], 'max_len': 103,
                         'messages': [{'role': 'assistant', 'content': 'inspect evidence'}]})
    assert report['max_tokens'] == 100
    assert report['output_tokens'] is None
    assert report['response_tail'] is None
    assert not report['output_recorded']
    assert report['last_visible_messages'][0]['content_tail'] == 'inspect evidence'


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


def test_diagnostics_keeps_earlier_errors_including_branch_requests():
    row = dict(source_index=6, status='ok', task_reward=0, is_finish=True)
    rejected = {'action_protocol_error': 'truncated JSON', 'max_tokens': 10, 'output_ids': [1] * 10}
    trajectory = {'model_contexts': [rejected, {}, {}],
                  'branch_model_contexts': {'branch': [rejected, {}, {}]}}
    errors = task_diagnostics(row, trajectory)['action_protocol_error_events']
    assert [e['agent'] for e in errors] == ['main', 'branch']
    assert all(e['max_tokens'] == 10 and e['output_tokens'] == 10 for e in errors)
