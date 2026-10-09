import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scripts.eval_bcp_qwen38 import config_for, select_indices
from scripts.prepare_agentfold_retry import prepare


def source_run(tmp_path):
    source = tmp_path / 'original'
    source.mkdir()
    data = tmp_path / 'original data.parquet'
    data.write_bytes(b'original dataset bytes')
    indices = [0, 11, 28, 70, 114]
    config = config_for(SimpleNamespace(method='agentfold', memory_mode='repaired'))
    manifest = dict(method='agentfold', benchmark='bcp', model='Qwen/Qwen3.5-9B',
                    model_path='/cached/original model', judge_model='original-judge',
                    data_path=str(data), source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                    indices=indices, config=OmegaConf.to_container(config), seed=42, commit='old-sha')
    for rank in range(3):
        (source / f'manifest-{rank}.json').write_text(json.dumps(dict(manifest, rank=rank)))
        rows = []
        for index in indices[rank::3]:
            failed = index in [11, 114]
            reason = 'invalid_tool_limit' if failed else 'finish'
            rows.append(dict(source_index=index, task_reward=int(not failed), is_finish=not failed,
                             status='ok', env_stats={'hit_format_retry_limit': int(failed)}))
            (source / f'trajectory-{index}.json').write_text(json.dumps({'termination_reason': reason}))
        (source / f'results-{rank}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    return source, data


def test_only_failures_selected_original_ids_data_and_seed_preserved(tmp_path):
    source, data = source_run(tmp_path)
    snapshot = {p.name: p.read_bytes() for p in source.iterdir()}
    target = tmp_path / 'retry'
    plan = prepare(source, target)
    assert plan['indices'] == [11, 114] and plan['count'] == 2 and plan['original_size'] == 5
    assert 'diagnostic' in plan['scope']
    selected = select_indices(150, -1, 42, target / 'indices.json')
    assert selected == [11, 114]
    assert sorted(sum((selected[rank::3] for rank in range(3)), [])) == selected
    env = (target / 'retry-env.sh').read_text()
    assert 'export SEED=42\n' in env
    assert 'original data.parquet' in env and 'original model' in env
    assert 'export JUDGE_MODEL=original-judge\n' in env
    assert not (target / 'run').exists()  # The exclusive launcher owns this leaf.
    assert {p.name: p.read_bytes() for p in source.iterdir()} == snapshot
    with pytest.raises(FileExistsError):
        prepare(source, target)


def test_retry_rejects_changed_original_data(tmp_path):
    source, data = source_run(tmp_path)
    data.write_bytes(b'reordered dataset')
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare(source, tmp_path / 'retry')


def test_retry_requires_result_and_trajectory_agreement(tmp_path):
    source, _ = source_run(tmp_path)
    (source / 'trajectory-11.json').write_text('{"termination_reason":"finish"}')
    with pytest.raises(ValueError, match='mismatch for 11'):
        prepare(source, tmp_path / 'retry')


@pytest.mark.parametrize('value', [[], [1, 1], [True], [-1], [150], [1.5], {'indices': [1]}])
def test_explicit_selection_rejects_ambiguous_ids(tmp_path, value):
    path = tmp_path / 'indices.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='unique in-range'):
        select_indices(150, -1, 42, path)


def test_explicit_selection_cannot_mix_with_random_sampling(tmp_path):
    path = tmp_path / 'indices.json'
    path.write_text('[114,11]')
    with pytest.raises(ValueError, match='samples=-1'):
        select_indices(150, 8, 42, path)
    assert select_indices(150, -1, 42, path) == [11, 114]
