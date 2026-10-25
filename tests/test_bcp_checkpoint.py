import json
from pathlib import Path

import pytest

from scripts.resolve_bcp_checkpoint import resolve


MODEL = 'Qwen/Qwen3.8-27B'


def checkpoint(root, revision='abc', model=MODEL):
    path = root / 'hf_cache' / 'hub' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision
    path.mkdir(parents=True)
    (path / 'config.json').write_text('{}')
    (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'part.safetensors'}}))
    (path / 'part.safetensors').write_bytes(b'weights')
    return path.resolve()


def test_snapshot_without_ref_and_cached_revision(tmp_path):
    first = checkpoint(tmp_path)
    assert resolve(tmp_path, MODEL) == first
    second = checkpoint(tmp_path, 'def')
    with pytest.raises(ValueError, match='Multiple complete snapshots'):
        resolve(tmp_path, MODEL)
    refs = first.parent.parent / 'refs'
    refs.mkdir()
    (refs / 'main').write_text('def\n')
    assert resolve(tmp_path, MODEL) == second
    assert resolve(tmp_path, MODEL, str(first)) == first
    (second / 'part.safetensors').unlink()
    with pytest.raises(ValueError, match='Incomplete checkpoint'):
        resolve(tmp_path, MODEL)


def test_no_silent_model_switch(tmp_path):
    other = checkpoint(tmp_path, model='Qwen/Qwen3.5-9B')
    with pytest.raises(ValueError, match='Cached repository directories.*Qwen/Qwen3.5-9B'):
        resolve(tmp_path, MODEL)
    with pytest.raises(ValueError, match='MODEL_PATH belongs to'):
        resolve(tmp_path, MODEL, str(other))


def test_explicit_path_and_incomplete_weights(tmp_path):
    model = checkpoint(tmp_path)
    with pytest.raises(ValueError, match='under SCRATCH'):
        resolve(tmp_path / 'other', MODEL, str(model))
    (model / 'part.safetensors').write_bytes(b'')
    with pytest.raises(ValueError, match='Incomplete checkpoint'):
        resolve(tmp_path, MODEL, str(model))
    with pytest.raises(ValueError, match='No complete local checkpoint'):
        resolve(tmp_path, MODEL)


def test_smoke_preserves_explicit_checkpoint(tmp_path):
    import os
    import shutil
    import subprocess
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash required')
    repo = Path(__file__).resolve().parents[1]
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    wrapper = scripts / 'smoke_graph_retry_ab_4node_idev.sh'
    shutil.copyfile(repo / 'scripts' / wrapper.name, wrapper)
    (scripts / 'eval_bcp_qwen38_4node_idev.sh').write_text('printf "%s|%s" "$MODEL_ID" "$MODEL_PATH" > "$PROJECT_ROOT/observed"\nexit 9\n')
    env = dict(os.environ, SCRATCH=tmp_path.as_posix(), SLURM_JOB_ID='123', SLURM_JOB_NODELIST='nodes', MODEL_PATH='/scratch/custom checkpoint')
    env.pop('MODEL_ID', None)
    # Source the actual wrapper with stub allocation/Git checks; stop at launcher.
    result = subprocess.run([bash, '-c', 'scontrol() { printf "a\\nb\\nc\\nd\\n"; }; git() { return 0; }; source "$1"', 'test', wrapper.as_posix()],
                            env=env, capture_output=True, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 9, result.stderr
    assert (tmp_path / 'observed').read_text() == 'Qwen/Qwen3.8-27B|/scratch/custom checkpoint'
