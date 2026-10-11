"""Read saved BC-P graph decisions and stop reasons; no performance/pass gate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_records import read_evaluation


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
        'last_model_turns': {agent: [
            {'phase': r.get('phase'), 'max_tokens': r.get('max_tokens'),
             'output_tokens': len(r.get('output_ids', [])),
             'response_tail': str(r.get('response', ''))[-1200:]}
            for r in turns[-2:]] for agent, turns in records.items()},
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
