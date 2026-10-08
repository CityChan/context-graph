"""Exercise the submission script with real Git worktrees and a fake scheduler."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('fail_second', [False, True])
def test_full_pair_pins_inputs_and_preserves_partial_submission(tmp_path, fail_second):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash required')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    def run(args, **kwargs):
        return subprocess.run(args, capture_output=True, text=True, timeout=40,
                              creationflags=flags, **kwargs)
    source = Path(__file__).resolve().parents[1]
    project = tmp_path / 'project with spaces'
    (project / 'scripts').mkdir(parents=True)
    script = 'submit_bcp_agentfold_supo_full.sh'
    shutil.copyfile(source / 'scripts' / script, project / 'scripts' / script)
    for method in ('supo', 'agentfold'):
        name = f'eval_bcp_{method}_qwen35_9b_4node.sbatch'
        shutil.copyfile(source / 'scripts' / name, project / 'scripts' / name)
    for args in (['init'], ['add', 'scripts'], ['-c', 'user.name=Test', '-c',
                  'user.email=test@example.invalid', 'commit', '-m', 'fixture']):
        result = run(['git', '-C', str(project), *args])
        assert result.returncode == 0, result.stderr
    (project / 'data').mkdir()
    (project / 'data/bc_test.parquet').write_bytes(b'pinned dataset fixture')
    stub = tmp_path / 'bin'
    stub.mkdir()
    scheduler = stub / 'sbatch'
    scheduler.write_text('''#!/bin/bash
set -eu
[[ "$SAMPLES" == -1 && "$WORKERS" == 1 && "$SEED" == 42 && "$BENCHMARK" == bcp ]]
[[ "$MODEL_MAX_LEN" == 32768 && "$SUPO_MAX_SUMMARIES" == 2 && "$MEMORY_MODE" == repaired ]]
[[ -z ${EXPECTED_JOB_ID:-} && ! -e "$RUN_ROOT" ]]
[[ -f "$DATA_PATH" && -f "$PROJECT_ROOT/.git" ]]
printf '%s\\n' "$*" >> "$TEST_LOG"
if [[ "$*" == *eval_bcp_supo_* ]]; then
  [[ "$FAIL_SECOND" == 0 ]] || exit 7
  echo 702
else
  echo 701
fi
''', encoding='utf-8')
    scheduler.chmod(0o755)
    env = dict(os.environ, PROJECT_ROOT=project.as_posix(),
               PATH=str(stub) + os.pathsep + os.environ['PATH'],
               TEST_LOG=(tmp_path / 'calls').as_posix(), FAIL_SECOND=str(int(fail_second)),
               SAMPLES='3', WORKERS='9', RUN_ROOT='must-not-reuse', EXPECTED_JOB_ID='old')
    env.pop('OUTPUT_BASE', None)
    env.pop('DATA_PATH', None)
    result = run([bash, str(project / 'scripts' / script)], env=env)
    assert result.returncode == int(fail_second), result.stdout + result.stderr
    submission, = (project / 'output').glob('bcp-agentfold-supo-full-*')
    assert (submission / 'bc_test.parquet').read_bytes() == b'pinned dataset fixture'
    assert (submission / 'data.sha256').is_file()
    rows = (submission / 'jobs.tsv').read_text().splitlines()
    assert len(rows) == (2 if fail_second else 3)
    assert rows[1].startswith('agentfold\t701\t')
    if not fail_second:
        assert rows[2].startswith('supo\t702\t')
    else:
        assert 'earlier jobs remain active' in result.stderr
    calls = (tmp_path / 'calls').read_text().splitlines()
    assert len(calls) == 2
    assert all('--time=06:00:00' in call and '--nodes=4' in call for call in calls)
    assert (submission / 'commit.txt').read_text().strip() == run(
        ['git', '-C', str(submission / 'code'), 'rev-parse', 'HEAD']).stdout.strip()
