"""Read saved BC-P graph decisions and stop reasons; no performance/pass gate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_records import read_evaluation


def model_turn(record):
    # Older ContextGraph traces store inputs only, unlike the memory baseline.
    # Missing outputs are unknown, not zero-token/empty model generations.
    maximum = record.get('max_tokens')
    if maximum is None and 'max_len' in record and 'input_ids' in record:
        maximum = max(record['max_len'] - len(record['input_ids']), 0)
    return {'max_tokens': maximum,
            'output_tokens': len(record['output_ids']) if 'output_ids' in record else None,
            'response_tail': str(record['response'])[-1200:] if 'response' in record else None,
            'output_recorded': 'response' in record,
            'action_protocol_error': record.get('action_protocol_error'),
            'dispatched_action_tail': str(record.get('dispatched_action', ''))[-1200:],
            'last_visible_messages': [{'role': m.get('role'), 'content_tail': str(m.get('content', ''))[-1200:]}
                                      for m in record.get('messages', [])[-2:]]}


def task_diagnostics(row, trajectory):
    stats = trajectory.get('env_stats', {})
    trace = trajectory.get('graph_trace', {})
    events = trace.get('events', [])
    explicit = [e for e in events if e.get('source') == 'model'
                and e.get('decision_context', {}).get('mode') == 'explicit_tool']
    # Environment observations and temporal/delegation edges are bookkeeping,
    # so report them separately from heuristic maintenance and model decisions.
    event_counts = Counter(f"{e.get('source')}:{e.get('op')}" for e in events)
    records = {'main': trajectory.get('model_contexts', [])}
    records.update(trajectory.get('branch_model_contexts', {}))
    return {
        'source_index': row['source_index'], 'status': row['status'],
        'task_reward': row['task_reward'], 'is_finish': row['is_finish'],
        'stop': trajectory.get('termination_reason'), 'stats': stats,
        'trace_present': bool(trace), 'events_by_source_and_op': dict(event_counts),
        'explicit_tool_attempts': dict(Counter(e['op'] for e in explicit)),
        'explicit_tool_successes': dict(Counter(e['op'] for e in explicit if e.get('success'))),
        'last_graph_errors': [{k: e.get(k) for k in ('turn_id', 'agent_name', 'op', 'args', 'error')}
                              for e in explicit if not e.get('success')][-3:],
        'last_model_turns': {agent: [model_turn(r) for r in turns[-2:]] for agent, turns in records.items()},
        'action_protocol_error_events': [
            {'agent': agent, 'request_index': index, **model_turn(record)}
            for agent, turns in records.items() for index, record in enumerate(turns)
            if record.get('action_protocol_error')],
        'saved_transcript_tail': trajectory.get('message_str', '')[-4000:],
    }


def diagnose(root):
    root = Path(root)
    manifests, rows, summary = read_evaluation(root)
    first = manifests[0]
    if first['method'] != 'contextgraph':
        raise ValueError('Expected a contextgraph evaluation')
    plugin = first['config']['actor_rollout_ref']['rollout']['plugin']
    fields = ('graph_action_protocol', 'structured_graph_controller',
              'controller_owned_tool_formatting', 'auto_prune_max_active',
              'auto_bind_branch_edges', 'consolidation_interval',
              'initial_consolidation_turn', 'enable_summary', 'enable_retrieval_memory')
    tasks = []
    for row in rows:
        path = root / f"trajectory-{row['source_index']}.json"
        if not path.exists():
            tasks.append({'source_index': row['source_index'], 'status': row['status'],
                          'trajectory_missing': True, 'error': row.get('error')})
            continue
        tasks.append(task_diagnostics(row, json.loads(path.read_text(encoding='utf-8'))))
    return {'scope': 'saved protocol and decision diagnostics; no pass or accuracy gate',
            'commit': first['commit'], 'summary': summary,
            'graph_config': {k: plugin.get(k) for k in fields}, 'tasks': tasks}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.output), indent=2, ensure_ascii=False))
