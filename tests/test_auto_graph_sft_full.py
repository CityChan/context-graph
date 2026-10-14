import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.generate_bcp_graph_sft import save_collection_manifest


def contract():
    return dict(commit='fixed-code', settings=dict(transport='vllm', teacher_url='http://node1:8000/v1',
                teacher_model='local-model', samples=-1, candidates=2), indices=[1, 2],
                source_sha256='train', heldout_sha256='heldout',
                local_teacher_checkpoint={'path': '/cache/snapshot', 'metadata_sha256': {'config.json': 'sha'}})


def test_resume_local_endpoint_preserves_manifest_and_records_transport(tmp_path):
    original = contract()
    save_collection_manifest(tmp_path, original)
    contents = (tmp_path / 'manifest.json').read_bytes()
    resumed = copy.deepcopy(original)
    resumed['settings']['teacher_url'] = 'http://node2:9000/v1'
    save_collection_manifest(tmp_path, resumed)
    assert (tmp_path / 'manifest.json').read_bytes() == contents
    sessions = [json.loads(s) for s in (tmp_path / 'collection-sessions.jsonl').read_text().splitlines()]
    assert sessions[-1]['teacher_url'] == resumed['settings']['teacher_url']
    assert sessions[0]['source_manifest_sha256'] == sessions[1]['source_manifest_sha256']


@pytest.mark.parametrize('change', ['commit', 'data', 'checkpoint', 'model', 'candidates', 'api'])
def test_resume_rejects_changed_contract(tmp_path, change):
    original = contract()
    if change == 'api':
        original['settings']['transport'] = 'deepseek'
        original['local_teacher_checkpoint'] = None
    save_collection_manifest(tmp_path, original)
    resumed = copy.deepcopy(original)
    resumed['settings']['teacher_url'] = 'http://node2:9000/v1'
    if change == 'commit':
        resumed['commit'] = 'different-code'
    elif change == 'data':
        resumed['source_sha256'] = 'different-data'
    elif change == 'checkpoint':
        resumed['local_teacher_checkpoint']['metadata_sha256']['config.json'] = 'different'
    elif change in ('model', 'candidates'):
        resumed['settings']['teacher_model' if change == 'model' else 'candidates'] = 'different'
    with pytest.raises(ValueError, match='Resume manifest mismatch'):
        save_collection_manifest(tmp_path, resumed)
    assert len((tmp_path / 'collection-sessions.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('busy', [False, True])
def test_full_wrapper_forces_full_collection_and_stable_resume_directory(tmp_path, busy):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash required')
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / 'scripts/generate_scienceworld_auto_sft_full_4node_idev.sh'
    shutil.copy(source, scripts / source.name)
    mocks = tmp_path / 'mocks.sh'
    mocks.write_text('git() { if [[ "$*" == *--show-toplevel* ]]; then echo "$TEST_ROOT"; else echo stablecommit; fi; }\nflock() { [[ "$LOCK_BUSY" == 0 ]]; }\n', newline='\n')
    (scripts / 'generate_scienceworld_graph_sft_deepseek_4node_idev.sh').write_text(
        'set -eu\n[[ "$SAMPLES" == -1 && "$START_INDEX" == 0 && "$CANDIDATES" == 2 && "$WORKERS" == 1 ]]\n'
        '[[ "$SFT_CURATION_SMOKE" == 0 && "$SFT_REQUIRE_COMPLETE" == 1 && "$SFT_GRAPH_PROTOCOL" == controller ]]\n'
        '[[ -z ${DRY_RUN:-} && -z ${ARRAY_CHUNK_MODE:-} ]]\n'
        'printf "%s\\n" "$RUN_DIR" >> "$TEST_ROOT/dispatched"\n', newline='\n')
    env = dict(os.environ, TEST_ROOT=tmp_path.as_posix(), SCRATCH=tmp_path.as_posix(),
               BASH_ENV=mocks.as_posix(), SLURM_JOB_ID='first', LOCK_BUSY=str(int(busy)),
               SAMPLES='3', CANDIDATES='2', START_INDEX='20', RUN_DIR='/stale/run',
               ARTIFACT_ROOT='/stale/root', SFT_CURATION_SMOKE='1', DRY_RUN='1', ARRAY_CHUNK_MODE='1')
    env.pop('AUTO_SFT_RUN_DIR', None)
    for job in ('first', 'second'):
        env['SLURM_JOB_ID'] = job
        result = subprocess.run([bash, str(scripts / source.name)], env=env, capture_output=True,
                                text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        assert result.returncode == (2 if busy else 0), result.stdout + result.stderr
    if busy:
        assert not (tmp_path / 'dispatched').exists()
    else:
        runs = (tmp_path / 'dispatched').read_text().splitlines()
        assert runs[0] == runs[1]
        assert Path(runs[0]).name == 'stablecommit'
