import json
import pytest
from scripts.audit_bcp_pair import audit


def runs(tmp_path):
    roots = []
    for method, scores in [('foldagent', [1, 0, 0]), ('contextgraph', [0, 1, 0])]:
        root = tmp_path / method
        root.mkdir()
        roots.append(root)
        for rank, score in enumerate(scores):
            manifest = dict(method=method, source_sha256='data', indices=[0, 1, 2],
                            commit='commit', model_path='checkpoint', seed=42,
                            judge_model='judge', config={'workflow': method})
            (root / f'manifest-{rank}.json').write_text(json.dumps(manifest))
            row = dict(source_index=rank, task_id=str(rank), task_reward=score,
                       is_finish=rank != 2, status='ok', env_stats={'search': rank})
            (root / f'results-{rank}.jsonl').write_text(json.dumps(row)+'\n')
    return roots


def test_pairs_by_identity_and_reports_missing_evidence(tmp_path):
    report = audit(*runs(tmp_path))
    assert report['groups'] == dict(both_correct=[], foldagent_only=[0], contextgraph_only=[1], both_wrong=[2])
    assert set(report['cases']) == {'0', '1', '2'}
    assert report['statistics']['all']['foldagent']['metrics']['forced_finish']['mean'] is None
    assert report['cases']['0']['foldagent']['judge_audit_available'] is False
    assert report['cases']['0']['foldagent']['request_count'] is None


def test_rejects_mismatched_task_and_incomplete_run(tmp_path):
    fold, graph = runs(tmp_path)
    path = graph / 'results-0.jsonl'
    row = json.loads(path.read_text())
    row['task_id'] = 'different'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='identity'):
        audit(fold, graph)
    path.write_text('')
    with pytest.raises(ValueError, match='Incomplete'):
        audit(fold, graph)
