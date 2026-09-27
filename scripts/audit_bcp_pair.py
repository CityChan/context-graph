"""Read-only paired audit of saved Qwen3.8 BC-P runs; no model/API calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean


METRICS = (
    'search', 'open_page', 'session_time', 'total_token', 'main_len', 'main_turn',
    'natural_finish', 'finalizer_attempted', 'forced_finish', 'pre_finalize_token_limit',
    'hit_token_limit', 'hit_max_turn', 'hit_timeout', 'observation_budget_truncations',
    'observation_budget_skips', 'is_branch', 'graph_explicit_ops', 'consol_attempts',
    'consol_ops', 'consol_controller_errors', 'memory_retrieval_calls',
    'memory_retrieval_evidence_tokens', 'memory_n_archives',
)
MATCH_FIELDS = ('source_sha256', 'indices', 'commit', 'model_path', 'seed', 'judge_model')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_run(root, method):
    manifests = [read_json(root / f'manifest-{rank}.json') for rank in range(3)]
    manifest = manifests[0]
    for other in manifests:
        for field in (*MATCH_FIELDS, 'method', 'config'):
            if other.get(field) != manifest.get(field):
                raise ValueError(f'{root}: shard mismatch in {field}')
    if manifest['method'] != method:
        raise ValueError(f'Expected {method}, got {manifest["method"]}')
    records = {}
    for rank in range(3):
        for line in (root / f'results-{rank}.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            index = row['source_index']
            if index in records:
                raise ValueError(f'Duplicate row {index} in {root}')
            if row['status'] != 'ok' or row.get('env_stats', {}).get('judge_parse_failure', 0):
                raise ValueError(f'Execution/judge failure in {root}, row {index}')
            if row['task_reward'] not in (0, 1):
                raise ValueError(f'Non-binary score in {root}, row {index}')
            records[index] = row
    if set(records) != set(manifest['indices']):
        raise ValueError(f'Incomplete/unexpected rows in {root}')
    return manifest, records


def config_differences(a, b, prefix=''):
    differences = {}
    for key in sorted(set(a) | set(b)):
        path = f'{prefix}.{key}' if prefix else key
        left, right = a.get(key), b.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            differences.update(config_differences(left, right, path))
        elif left != right:
            differences[path] = {'foldagent': left, 'contextgraph': right}
    return differences


def aggregate(rows):
    result = {'count': len(rows), 'correct': sum(r['task_reward'] for r in rows),
              'finished': sum(bool(r['is_finish']) for r in rows)}
    result['metrics'] = {}
    for key in METRICS:
        values = [r['env_stats'][key] for r in rows if isinstance(r.get('env_stats', {}).get(key), (int, float))]
        result['metrics'][key] = {'observed': len(values), 'mean': mean(values) if values else None,
                                  'positive': sum(v > 0 for v in values)}
    searches = [r['env_stats']['search'] for r in rows if 'search' in r.get('env_stats', {})]
    result['zero_search'] = {'observed': len(searches), 'count': searches.count(0)}
    return result


def case_evidence(root, index, row):
    path = root / f'trajectory-{index}.json'
    trajectory = read_json(path) if path.exists() else {}
    audits = trajectory.get('judge_audit')
    messages = trajectory.get('messages', [])
    calls, lengths, generated = Counter(), [], 0
    request_path = root / f'requests-{index}.jsonl'
    if request_path.exists():
        with request_path.open(encoding='utf-8') as handle:
            for line in handle:
                request = json.loads(line)
                calls[str(request.get('finish_reason'))] += 1
                lengths.append(len(request.get('input_ids', [])))
                generated += len(request.get('output_ids', []))
    return {'result': row, 'trajectory_file': str(path), 'trajectory_present': path.exists(),
            'termination_reason': trajectory.get('termination_reason'),
            'num_branches': trajectory.get('num_branches'),
            # Missing FoldAgent judge audits must not be interpreted as zero blanks.
            'judge_audit_available': isinstance(audits, list) and bool(audits),
            'judge_audit': audits,
            'request_count': sum(calls.values()) if request_path.exists() else None,
            'request_finish_reasons': dict(calls),
            'max_input_tokens': max(lengths) if lengths else None,
            'generated_tokens_all_requests': generated if request_path.exists() else None,
            'tail_messages': [{'role': m.get('role'), 'content': str(m.get('content', ''))[-4000:],
                               'excerpt_truncated': len(str(m.get('content', ''))) > 4000}
                              for m in messages[-6:]]}


def audit(fold_root, graph_root):
    fm, fold = load_run(fold_root, 'foldagent')
    gm, graph = load_run(graph_root, 'contextgraph')
    for field in MATCH_FIELDS:
        if fm.get(field) != gm.get(field):
            raise ValueError(f'Cross-run mismatch in {field}')
    groups = {'both_correct': [], 'foldagent_only': [], 'contextgraph_only': [], 'both_wrong': []}
    for index in sorted(fold):
        if fold[index]['task_id'] != graph[index]['task_id']:
            raise ValueError(f'Task identity mismatch at row {index}')
        key = {(1, 1): 'both_correct', (1, 0): 'foldagent_only',
               (0, 1): 'contextgraph_only', (0, 0): 'both_wrong'}[
                   (fold[index]['task_reward'], graph[index]['task_reward'])]
        groups[key].append(index)
    stats = {name: {method: aggregate([rows[i] for i in indices])
                    for method, rows in [('foldagent', fold), ('contextgraph', graph)]}
             for name, indices in {'all': sorted(fold), **groups}.items()}
    cases = {}
    # Capture all discordant and unfinished cases, not just hand-picked examples.
    selected = set(groups['foldagent_only'] + groups['contextgraph_only'])
    selected.update(i for i in fold if not fold[i]['is_finish'] or not graph[i]['is_finish'])
    for index in sorted(selected):
        cases[str(index)] = {'foldagent': case_evidence(fold_root, index, fold[index]),
                             'contextgraph': case_evidence(graph_root, index, graph[index])}
    return {'provenance': {f: fm.get(f) for f in MATCH_FIELDS},
            'config_differences': config_differences(fm['config'], gm['config']),
            'groups': groups, 'statistics': stats, 'cases': cases,
            'limits': 'Paired associations, not causal ablations. Missing metrics are unknown. '
                      'Saved chats may contain rewritten history; token records are raw requests. '
                      'Token totals include controller and finalizer calls. Tail excerpts are incomplete.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--foldagent', type=Path, required=True)
    parser.add_argument('--contextgraph', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.foldagent, args.contextgraph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'config_differences': report['config_differences'],
                      'groups': report['groups']}, ensure_ascii=False, indent=2))
    for group, methods in report['statistics'].items():
        for method, stats in methods.items():
            fields = {k: stats[k] for k in ('count', 'correct', 'finished', 'zero_search')}
            fields['metrics'] = {k: stats['metrics'][k] for k in (
                'forced_finish', 'pre_finalize_token_limit', 'search', 'session_time', 'total_token')}
            print(group, method, json.dumps(fields))
    print(f'Full case evidence: {args.output}')


if __name__ == '__main__':
    main()
