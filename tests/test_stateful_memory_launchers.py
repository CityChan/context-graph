import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts.audit_stateful_memory_smoke import audit


@pytest.mark.parametrize('benchmark', ['swe-lite', 'scienceworld', 'discoveryworld'])
@pytest.mark.parametrize('method', ['memobrain', 'amem'])
def test_graph_native_audit_requires_real_memory_protocol_and_accounting(tmp_path, benchmark, method):
    protocol = ('swe' if benchmark == 'swe-lite' else benchmark) + '_v1'
    trajectory = dict(termination_reason='finish', memory_audit=[], **{method: dict(task_protocol=protocol)},
        env_stats=dict(environment_steps=6, python_exec=6, memory_updates=5, memory_nodes=4,
                       memobrain_folds=1, amem_evolutions=2, embedding_calls=10, memory_retrieval_calls=6,
                       generated_tokens=50, observation_tokens=40, instruction_tokens=10, main_len=100,
                       actor_requests=6, memory_requests=5, model_requests=11, invalid_memory=0))
    for i in range(2):
        part = tmp_path / (f'pair-{i}' if benchmark == 'swe-lite' else f'{method}-{i}')
        attempt = part / 'instances/task/attempt-1'
        attempt.mkdir(parents=True)
        (part / 'summary.json').write_text(json.dumps(dict(selected=1, pending=0)))
        (attempt.parent / 'result.json').write_text(json.dumps(dict(status='graded', attempt='attempt-1')))
        target = attempt / ('generation/instances/task/trajectory.json' if benchmark == 'swe-lite' else 'trajectory.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(trajectory))
    assert audit(tmp_path, benchmark, method, 2)['passed']
    for field, value, check in [('invalid_memory', 1, 'memory_format_valid'),
                               ('main_len', 99, 'budget_accounting'),
                               ('model_requests', 12, 'budget_accounting')]:
        old = trajectory['env_stats'][field]
        trajectory['env_stats'][field] = value
        target.write_text(json.dumps(trajectory))
        report = audit(tmp_path, benchmark, method, 2)
        assert not report['passed'] and not report['checks'][check]
        trajectory['env_stats'][field] = old
    trajectory[method]['task_protocol'] = 'search'
    target.write_text(json.dumps(trajectory))
    assert not audit(tmp_path, benchmark, method, 2)['checks']['native_protocol']


def test_discoveryworld_summary_keeps_format_failures_distinct_from_infrastructure(tmp_path):
    from scripts.eval_agent_benchmarks import summary, task_key
    path = tmp_path / 'instances' / task_key('test')
    path.mkdir(parents=True)
    (path / 'result.json').write_text(json.dumps(dict(status='graded', score=25, success=False,
                                                    termination_reason='invalid_summary')))
    report = summary(tmp_path, ['test'], 'discoveryworld')
    assert report['format_failures'] == 1 and report['infrastructure_errors'] == 0
    assert report['mean_score'] == 25


@pytest.mark.parametrize('benchmark', ['scienceworld', 'discoveryworld'])
def test_amem_setup_records_are_not_shards_and_incomplete_runs_still_fail(tmp_path, benchmark):
    for name in ('amem-native-python.txt', 'amem-native-environment.txt'):
        (tmp_path / name).write_text('setup provenance')
    for i in range(2):
        part = tmp_path / f'amem-{i}'
        attempt = part / f'instances/task-{i}/attempt-1'
        attempt.mkdir(parents=True)
        (part / 'summary.json').write_text(json.dumps(dict(selected=1, pending=0)))
        (attempt.parent / 'result.json').write_text(json.dumps(dict(status='graded', attempt='attempt-1')))
        trajectory = dict(termination_reason='max_turn', amem=dict(task_protocol=benchmark + '_v1'),
            env_stats=dict(environment_steps=34, actor_requests=34, memory_requests=66, model_requests=100,
                           memory_updates=66, memory_nodes=34, amem_evolutions=32, invalid_memory=0,
                           embedding_calls=100, memory_retrieval_calls=34, generated_tokens=10,
                           observation_tokens=20, instruction_tokens=30, main_len=60,
                           environment_score=0, completed=0, hit_max_turn=1))
        (attempt / 'trajectory.json').write_text(json.dumps(trajectory))
    report = audit(tmp_path, benchmark, 'amem', 2)
    assert report['passed']
    assert all(t['stop'] == 'max_turn' and t['stats']['environment_score'] == 0 for t in report['tasks'])
    assert not audit(tmp_path, benchmark, 'amem', 3)['checks']['complete']

    summary = part / 'summary.json'
    summary.unlink()
    assert not audit(tmp_path, benchmark, 'amem', 2)['checks']['complete']
    summary.write_text(json.dumps(dict(selected=1, pending=1)))
    assert not audit(tmp_path, benchmark, 'amem', 2)['checks']['complete']
    summary.write_text(json.dumps(dict(selected=1, pending=0)))
    extra = tmp_path / 'amem-2'
    extra.mkdir()
    assert not audit(tmp_path, benchmark, 'amem', 2)['checks']['complete']
    extra.rmdir()
    result = attempt.parent / 'result.json'
    result.unlink()
    assert not audit(tmp_path, benchmark, 'amem', 2)['checks']['complete']
    # Removing a shard must still fail even with the setup files present.
    shutil.rmtree(part)
    assert not audit(tmp_path, benchmark, 'amem', 2)['checks']['complete']


@pytest.mark.parametrize('benchmark', ['swe-lite', 'discoveryworld', 'scienceworld'])
@pytest.mark.parametrize('method', ['supo', 'agentfold'])
def test_smoke_audit_requires_grading_complete_and_memory(tmp_path, benchmark, method):
    for i in range(2):
        part = tmp_path / (f'pair-{i}' if benchmark == 'swe-lite' else f'{method}-{i}')
        attempt = part / 'instances/task/attempt-1'
        attempt.mkdir(parents=True)
        (part / 'summary.json').write_text(json.dumps(dict(selected=1, pending=0)))
        (attempt.parent / 'result.json').write_text(json.dumps(dict(status='graded', attempt='/old/location/attempt-1')))
        trajectory = dict(termination_reason='finish', env_stats=dict(environment_steps=2, python_exec=1,
                            agentfold_folds=1, summary_restarts=1), **{method: {'task_protocol': benchmark}})
        target = attempt / 'trajectory.json'
        if benchmark == 'swe-lite':
            target = attempt / 'generation/instances/task/trajectory.json'
            target.parent.mkdir(parents=True)
        target.write_text(json.dumps(trajectory if benchmark == 'swe-lite' else [trajectory]))
    assert audit(tmp_path, benchmark, method, 2)['passed']
    assert not audit(tmp_path, benchmark, method, 8)['passed']
    trajectory['termination_reason'] = 'invalid_summary'
    target.write_text(json.dumps(trajectory if benchmark == 'swe-lite' else [trajectory]))
    assert not audit(tmp_path, benchmark, method, 2)['passed']


@pytest.mark.parametrize('benchmark', ['swe-lite', 'discoveryworld', 'scienceworld'])
@pytest.mark.parametrize('method', ['supo', 'agentfold', 'memobrain', 'amem'])
def test_wrapper_dispatches_native_launchers_and_ignores_stale_roots(tmp_path, benchmark, method):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash required')
    source = Path(__file__).resolve().parents[1]
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    mocks = tmp_path / 'mocks.sh'
    mocks.write_text('git() { echo "$TEST_ROOT"; }\n', newline='\n')
    for name in ('eval_discoveryworld_qwen35_9b_4node.sbatch', 'eval_swe_lite_arm_4node.sbatch',
                 'eval_agent_benchmarks_qwen35_9b_4node.sbatch'):
        (scripts / name).write_text('set -eu\nif [[ "$1" == scienceworld ]]; then '
            '[[ "$BENCH_PROMPT_PROFILE" == focus_v3 && "$BENCH_MAX_STEPS" == 100 && '
            '"$BENCH_REQUIREMENTS" == requirements_agent_benchmarks.txt ]]; shift; fi\n'
            'printf "%s|%s|%s\\n" "$1" "${BENCH_RUN_DIR:-$SWE_RUN_DIR}" "$SAMPLES" > "$PROJECT_ROOT/dispatched"\n', newline='\n')
    python = tmp_path / 'env/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" > "$TEST_ROOT/audited"\n', newline='\n')
    python.chmod(0o755)
    env = dict(os.environ, BASH_ENV=mocks.as_posix(), TEST_ROOT=tmp_path.as_posix(),
               PROJECT_ROOT='/stale/worktree', SCRATCH=tmp_path.as_posix(), SLURM_JOB_ID='test123',
               BENCH_BASE_PYTHON=python.as_posix(), SWE_AGENT_ENV=(tmp_path / 'env').as_posix(),
               BENCH_DATA='/stale/data', BENCH_RUN_DIR='/stale/run', SWE_RUN_DIR='/stale/run', SAMPLES='2')
    # An unrelated BENCH_RUN_DIR must not affect SWE; inspect native output variable only.
    if benchmark == 'swe-lite':
        env.pop('BENCH_RUN_DIR')
    result = subprocess.run([bash, str(source / 'scripts/smoke_stateful_memory_qwen35_9b_4node_idev.sh'), benchmark, method],
                            env=env, capture_output=True, text=True, timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stdout + result.stderr
    selected, output, count = (tmp_path / 'dispatched').read_text().strip().split('|')
    assert selected == method and count == '2'
    assert Path(output).is_dir() and Path(output).parent == tmp_path / 'output'
    assert f'{benchmark} {method} --expected 2' in (tmp_path / 'audited').read_text()
