"""Check an existing curated export without a teacher, GPU, or split changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit_export(directory):
    import pandas as pd

    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    decisions = [json.loads(line) for line in (root / 'decisions.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]
    counts = manifest['counts']
    accepted = {d['episode'] for d in decisions if d.get('reason') is None}
    frames, file_hashes = {}, {}
    for role in ('actor', 'controller'):
        for split in ('train', 'validation'):
            name = role + '_' + split
            path = root / (name + '.parquet')
            if path.is_file():
                frames[name] = pd.read_parquet(path)
                file_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sizes = {name: len(frame) for name, frame in frames.items()}
    tasks = {split: set() for split in ('train', 'validation')}
    structural = {split: 0 for split in tasks}
    valid_rows = True
    for name, frame in frames.items():
        role, split = name.split('_')
        required = {'task_id', 'episode', 'target_role', 'target_action', 'tokens',
                    'task_reward', 'graph_trace_valid', 'train_last_response_only',
                    'graph_action_protocol', 'timing_source'}
        if not required.issubset(frame.columns):
            valid_rows = False
            continue
        tasks[split].update(frame['task_id'])
        valid_rows = valid_rows and bool(
            frame['episode'].isin(accepted).all()
            and frame['target_role'].eq(role).all()
            and frame['task_reward'].eq(1).all()
            and frame['graph_trace_valid'].eq(True).all()
            and frame['train_last_response_only'].eq(True).all()
            and frame['graph_action_protocol'].eq('controller').all()
            and frame['tokens'].between(1, manifest['settings']['max_length']).all())
        if role == 'controller':
            valid_rows = valid_rows and bool(frame['timing_source'].eq('framework').all())
            structural[split] += int(frame['target_action'].isin({'merge', 'prune', 'add_edge'}).sum())
    checks = {
        'collection_complete': counts.get('incomplete_episodes') == 0
            and counts.get('episodes_seen') == counts.get('planned_episodes') == len(decisions),
        'accepted_episodes_recorded': bool(accepted) and sum(d.get('reason') is None for d in decisions)
            == counts.get('accepted_episodes'),
        'export_counts_match': sizes == manifest['outputs'] and all(
            counts.get('rows:' + name) == size for name, size in sizes.items()),
        'curated_row_metadata_valid': valid_rows,
        'task_splits_disjoint': not (tasks['train'] & tasks['validation']),
        'actor_rows_present': sum(size for name, size in sizes.items() if name.startswith('actor_')) > 0,
        'structural_controller_rows_present': sum(structural.values()) > 0,
        'structural_counts_match': all(structural[s] == counts.get('structural_rows:' + s, 0) for s in tasks),
    }
    passed = all(checks.values())
    training_ready = passed and sizes.get('actor_train', 0) > 0 and structural['train'] > 0
    return {'scope': 'saved curated export consistency only; no new evidence review or model evaluation',
            'passed': passed, 'training_ready': training_ready, 'checks': checks,
            'accepted_episodes': counts.get('accepted_episodes', 0), 'outputs': sizes,
            'structural_rows': structural, 'parquet_sha256': file_hashes,
            'source': str(root.resolve()),
            'note': 'Fixed task splits preserved. Validation rows are not training rows.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    parser.add_argument('--require-training', action='store_true')
    args = parser.parse_args()
    report = audit_export(args.directory)
    print(json.dumps(report, indent=2))
    if not report['passed'] or (args.require_training and not report['training_ready']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
