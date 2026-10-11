import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('benchmark', ['bcp', 'discoveryworld'])
@pytest.mark.parametrize('protocol', ['controller', 'explicit_tools'])
def test_eval_manifest_config_matches_selected_protocol(benchmark, protocol):
    if benchmark == 'bcp':
        from scripts.eval_bcp_qwen38 import config_for
        config = config_for(SimpleNamespace(method='contextgraph', memory_mode='repaired', graph_action_protocol=protocol))
    else:
        from scripts.eval_agent_benchmarks import config_for
        config = config_for('discoveryworld', 'contextgraph', 65536, prompt_profile='discoveryworld_v1',
                            graph_action_protocol=protocol)
    plugin = config.actor_rollout_ref.rollout.plugin
    assert plugin.graph_action_protocol == protocol
    assert plugin.structured_graph_controller == (protocol == 'controller')
    assert plugin.auto_prune_max_active == (12 if protocol == 'controller' else 0)
    if protocol == 'explicit_tools':
        assert plugin.graph_decision_profile == 'graph_affordances_v2'
        assert not any([plugin.auto_bind_branch_edges, plugin.enable_summary, plugin.enable_retrieval_memory,
                        plugin.consolidation_interval, plugin.initial_consolidation_turn,
                        plugin.controller_owned_tool_formatting])


def test_reject_explicit_protocol_for_other_baselines():
    from scripts.eval_bcp_qwen38 import config_for
    with pytest.raises(ValueError, match='contextgraph'):
        config_for(SimpleNamespace(method='foldagent', memory_mode='repaired', graph_action_protocol='explicit_tools'))


def test_discoveryworld_parent_passes_protocol_to_real_worker_parser(monkeypatch, tmp_path):
    from scripts import eval_agent_benchmarks as module
    task = {'task_id': 'task-one'}
    args = SimpleNamespace(output=tmp_path, benchmark='discoveryworld', method='contextgraph',
        shard_index=0, shard_count=1, retry_errors=False, endpoint='http://unused', model_path='unused',
        context_length=65536, max_steps=100, memory_profile='repaired', prompt_profile='discoveryworld_v1',
        discoveryworld_observation_profile='compact_v1', graph_action_protocol='explicit_tools', task_timeout=30)
    monkeypatch.setattr(module, 'preflight', lambda args: ([task], {'protocol': args.graph_action_protocol}))
    monkeypatch.setattr(module, 'summary', lambda *a: {'infrastructure_errors': 0})
    received = []
    async def fake_generate(parsed, task, directory):
        received.append(parsed.graph_action_protocol)
        (directory / 'result.json').write_text(json.dumps({'status': 'graded', 'score': 0, 'success': False}))
    monkeypatch.setattr(module, 'generate', fake_generate)
    def worker(command, log, timeout):
        monkeypatch.setattr(sys, 'argv', command[2:])
        assert module.main() == 0
    monkeypatch.setattr(module, 'run_command', worker)
    assert module.run(args) == 0
    assert received == ['explicit_tools']
    assert json.loads((tmp_path / 'manifest.json').read_text())['protocol'] == 'explicit_tools'


@pytest.mark.parametrize('benchmark', ['bcp', 'discoveryworld'])
def test_real_shell_evaluator_role_forwards_protocol(tmp_path, benchmark):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    repo = Path(__file__).resolve().parents[1]
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    fake_python = binaries / 'python'
    fake_python.write_text('#!/bin/bash\nif [[ ${1:-} == -c ]]; then printf "%s\\n" "$FAKE_LIBRARY"; else printf "%s\\n" "$@" > "$CAPTURE"; fi\n', encoding='utf8')
    fake_python.chmod(0o755)
    library = tmp_path / 'fake.so'
    library.write_text('preload stub')
    conda = tmp_path / 'conda.sh'
    conda.write_text('conda() { :; }\nexport PATH="$(cd "$STUB_BIN" && pwd):$PATH"\n', encoding='utf8')
    # Capture the final exec before starting Python (or loading Linux Torch on Windows).
    startup = tmp_path / 'startup.sh'
    startup.write_text('exec() { printf "%s\\n" "$@" > "$CAPTURE"; exit 0; }\n', encoding='utf8')
    capture = tmp_path / 'arguments.txt'
    env = dict(os.environ, PROJECT_ROOT=repo.as_posix(), SCRATCH=tmp_path.as_posix(),
        SLURM_JOB_ID='test', CONDA_SH=conda.as_posix(), METHOD='contextgraph',
        MODEL_PATH='unused', RUN_ROOT=tmp_path.as_posix(), BENCHMARK=benchmark,
        BENCH_AGENT_ENV=tmp_path.as_posix(), BENCH_RUN_DIR=tmp_path.as_posix(), BENCH_DATA='unused',
        GRAPH_ACTION_PROTOCOL='explicit_tools', CAPTURE=capture.as_posix(), FAKE_LIBRARY=library.as_posix(),
        STUB_BIN=binaries.as_posix(), BASH_ENV=startup.as_posix())
    if benchmark == 'bcp':
        script, role = 'eval_bcp_qwen38_4node_idev.sh', ['_eval', 'http://unused', '0']
    else:
        script, role = 'eval_agent_benchmarks_qwen35_9b_4node.sbatch', ['_eval', 'contextgraph', 'http://unused', 'contextgraph-0', '0', '2']
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    result = subprocess.run([bash, (repo / 'scripts' / script).as_posix(), *role], env=env,
                            capture_output=True, text=True, timeout=30, **options)
    assert result.returncode == 0, result.stdout + result.stderr
    forwarded = capture.read_text().splitlines()
    assert forwarded[forwarded.index('--graph-action-protocol') + 1] == 'explicit_tools'
