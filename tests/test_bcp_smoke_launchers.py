"""Run real smoke wrappers in Bash with a stub of the exclusive-directory launcher."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize('method', ['supo', 'agentfold'])
@pytest.mark.parametrize('scenario', ['default', 'custom', 'existing', 'startup_failure', 'evaluation_failure'])
def test_smoke_directory_ownership_and_failure_audit(tmp_path, method, scenario):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash required for launcher integration test')
    repo = Path(__file__).resolve().parents[1]
    project = tmp_path / 'project with spaces'
    scripts = project / 'scripts'
    scripts.mkdir(parents=True)
    scratch = tmp_path / 'scratch with spaces'
    scratch.mkdir()
    conda = tmp_path / 'conda.sh'
    conda.write_text('conda() { :; }\npython() { printf "%s\\n" "$*" > "$RUN_ROOT/audit-called"; }\n')
    launcher = scripts / 'eval_bcp_qwen35_9b_4node_idev.sh'
    # Matches the actual launcher: the run leaf must not exist before mkdir.
    launcher.write_text('set -euo pipefail\nprintf "%s\\n" "$RUN_ROOT" > "$PROJECT_ROOT/launch-called"\n'
                        'mkdir -p "$(dirname "$RUN_ROOT")"\nmkdir "$RUN_ROOT"\n'
                        '[[ "$TEST_SCENARIO" != startup_failure ]] || exit 1\n'
                        'for rank in 0 1 2; do printf "{}" > "$RUN_ROOT/manifest-$rank.json"; '
                        ': > "$RUN_ROOT/results-$rank.jsonl"; done\n'
                        '[[ "$TEST_SCENARIO" != evaluation_failure ]]\n', encoding='utf-8')
    env = dict(os.environ, PROJECT_ROOT=project.as_posix(), SCRATCH=scratch.as_posix(),
               CONDA_SH=conda.as_posix(), SLURM_JOB_ID='test123', TEST_SCENARIO=scenario)
    env.pop('RUN_ROOT', None)
    if scenario in {'custom', 'existing'}:
        target = scratch / 'explicit run'
        env['RUN_ROOT'] = target.as_posix()
        if scenario == 'existing':
            target.mkdir()
            (target / 'sentinel').write_text('preserve me')
    result = subprocess.run([bash, str(repo / f'scripts/smoke_bcp_{method}_qwen35_9b_4node_idev.sh')],
                            env=env, capture_output=True, text=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if scenario == 'existing':
        assert result.returncode == 2 and 'already exists' in result.stdout
        assert not (project / 'launch-called').exists()
        assert (target / 'sentinel').read_text() == 'preserve me'
        return
    output = Path((project / 'launch-called').read_text().strip())
    assert output.is_dir()
    if scenario == 'default':
        assert output.name == 'run' and output.parent.parent == scratch
    if scenario == 'startup_failure':
        assert result.returncode == 1 and 'AUDIT_SKIPPED' in result.stdout
        assert not (output / 'audit-called').exists()
    else:
        assert result.returncode == (1 if scenario == 'evaluation_failure' else 0), result.stderr
        assert (output / 'audit-called').is_file()
    assert 'File exists' not in result.stderr and 'Traceback' not in result.stderr
