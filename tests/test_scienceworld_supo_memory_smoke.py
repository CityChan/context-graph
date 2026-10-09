import os
from pathlib import Path
import shutil
import subprocess

from omegaconf import OmegaConf
import pytest

from scripts.eval_agent_benchmarks import config_for


def test_memory_smoke_changes_only_threshold_and_scope():
    normal = config_for('scienceworld', 'supo', 65536, prompt_profile='focus_v3')
    smoke = config_for('scienceworld', 'supo', 65536, prompt_profile='focus_v3', memory_smoke=True)
    plugin = smoke.actor_rollout_ref.rollout.plugin
    assert plugin.supo_context_threshold == 4096
    assert plugin.evaluation_scope == 'memory_smoke_4k_not_benchmark'
    assert normal.actor_rollout_ref.rollout.plugin.supo_context_threshold == 32768
    plugin.supo_context_threshold = 32768
    del plugin.evaluation_scope
    assert OmegaConf.to_container(smoke) == OmegaConf.to_container(normal)


@pytest.mark.parametrize('benchmark,method', [('scienceworld', 'agentfold'), ('discoveryworld', 'supo')])
def test_diagnostic_rejects_other_protocols(benchmark, method):
    with pytest.raises(ValueError, match='only for ScienceWorld SUPO'):
        config_for(benchmark, method, 65536, memory_smoke=True)


def test_dedicated_entry_pins_two_tasks_and_scope(tmp_path):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash:
        pytest.skip('Bash required')
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / 'smoke_scienceworld_supo_memory_4node_idev.sh'
    shutil.copyfile(root / 'scripts' / script.name, script)
    (tmp_path / 'smoke_stateful_memory_qwen35_9b_4node_idev.sh').write_text(
        'printf "%s|%s|%s|%s" "$STATEFUL_MEMORY_SMOKE" "$SAMPLES" "$1" "$2"\n', newline='\n')
    result = subprocess.run([bash, script.as_posix()], env=dict(os.environ, SAMPLES='-1'),
                            capture_output=True, text=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stderr
    assert result.stdout == '1|2|scienceworld|supo'
