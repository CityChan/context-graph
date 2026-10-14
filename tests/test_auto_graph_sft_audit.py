import json

import pandas as pd
import pytest

from scripts.audit_auto_graph_sft import audit_export


def export_fixture(root, split='validation'):
    metadata = dict(task_id='task', episode='episodes/1/episode.json',
                    task_reward=1, graph_trace_valid=True, train_last_response_only=True,
                    tokens=100, graph_action_protocol='controller')
    outputs = {}
    for role, action in [('actor', 'action'), ('controller', 'merge')]:
        name = role + '_' + split
        pd.DataFrame([{**metadata, 'target_role': role, 'target_action': action,
                       'timing_source': 'framework' if role == 'controller' else 'model'}]).to_parquet(root / (name + '.parquet'))
        outputs[name] = 1
    counts = dict(planned_episodes=1, episodes_seen=1, incomplete_episodes=0, accepted_episodes=1)
    counts.update({'rows:' + k: v for k, v in outputs.items()})
    counts['structural_rows:' + split] = 1
    (root / 'manifest.json').write_text(json.dumps(dict(counts=counts, outputs=outputs, settings={'max_length': 32768})))
    (root / 'decisions.jsonl').write_text(json.dumps({'episode': metadata['episode'], 'reason': None}) + '\n')


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_audit_distinguishes_collection_from_training_readiness(tmp_path, split):
    export_fixture(tmp_path, split)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report = audit_export(tmp_path)
    assert report['passed'] and report['training_ready'] == (split == 'train')
    assert report['structural_rows'][split] == 1
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('failure', ['missing_file', 'count', 'unsolved', 'oversize', 'unreviewed', 'split_leak'])
def test_audit_rejects_inconsistent_exports(tmp_path, failure):
    export_fixture(tmp_path)
    path = tmp_path / 'controller_validation.parquet'
    frame = pd.read_parquet(path)
    if failure == 'missing_file':
        path.unlink()
    elif failure == 'count':
        pd.concat([frame, frame]).to_parquet(path)
    elif failure in ('unsolved', 'oversize', 'unreviewed'):
        key, value = {'unsolved': ('task_reward', 0), 'oversize': ('tokens', 32769),
                      'unreviewed': ('episode', 'unknown')}[failure]
        frame[key] = value
        frame.to_parquet(path)
    else:
        frame.to_parquet(tmp_path / 'controller_train.parquet')
        manifest = json.loads((tmp_path / 'manifest.json').read_text())
        manifest['outputs']['controller_train'] = 1
        manifest['counts'].update({'rows:controller_train': 1, 'structural_rows:train': 1})
        (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    report = audit_export(tmp_path)
    assert not report['passed'] and not report['training_ready']
    if failure == 'split_leak':
        assert not report['checks']['task_splits_disjoint']
