import asyncio
from types import SimpleNamespace

import pytest

from agents.action_contract import ActionContract
from agents.sft_teacher_guidance import AssistedMaintenance, teacher_guidance
from agents.utils import Agent
from tests.test_session_restart import Client, Tokenizer


def test_assistance_is_bounded_rejects_task_actions_and_preserves_sampled_policy():
    config = SimpleNamespace(prompt_length=8192, response_length=24576,
        plugin=SimpleNamespace(sft_teacher_profile='framework_assisted_v1',
                               graph_action_protocol='explicit_tools',
                               token_budget_mode='working_context', capture_model_contexts=True))
    raw = '{"tool":"action","arguments":{"command":"look around"}}'
    agent = Agent(Client([raw] * 8), [{'role': 'system', 'content': 'rules'},
                  {'role': 'user', 'content': 'experiment'}], Tokenizer(), config,
                  action_contract=ActionContract('ScienceWorld@real', None))
    for _ in range(7):
        agent.sft_maintenance.sampled({'tool': 'action'})
    agent.sft_maintenance.prepare(agent)
    assert agent.sft_maintenance.active is None
    agent.sft_maintenance.sampled({'tool': 'action'})
    for _ in range(6):
        wire = asyncio.run(agent.step())
        assert wire.startswith('[Action protocol rejected]')
        assert agent.chat[-1]['content'] == raw
        agent.append({'role': 'user', 'content': wire})
    assert len(agent.llm_client.calls) == 6
    assert all(call['max_len'] <= 32768 for _, call in agent.llm_client.calls)
    assert len([m for m in agent.chat if 'checkpoint 1;' in m['content']]) == 1
    with pytest.raises(RuntimeError, match='request limit'):
        asyncio.run(agent.step())
    assert len(agent.llm_client.calls) == 6


def test_pressure_trigger_and_real_compaction_result_control_resume():
    class Stub:
        config = SimpleNamespace(prompt_length=8192, response_length=24576)
        force_action_finish = False
        context_uid = 'branch-1'
        def context(self):
            return [0] * 18000
        def append(self, message):
            self.message = message
    agent = Stub()
    state = AssistedMaintenance()
    state.task_actions = 1
    state.prepare(agent)
    assert state.active is None
    state.task_actions = 2
    agent.force_action_finish = True
    state.prepare(agent)
    assert state.active is None  # Reserved branch return/final answer takes precedence.
    agent.force_action_finish = False
    state.prepare(agent)
    assert state.active['trigger'] == 'window_pressure'
    assert state.active['context_uid_at_trigger'] == 'branch-1'
    state.executed('merge', True, 18000, 18000)
    state.executed('graph_compact', False, 18000, 18000)
    state.executed('graph_compact', True, 18000, 19000)
    assert state.active
    state.executed('graph_compact', True, 18000, 8000)
    assert state.active is None and state.task_actions == 0


def test_assisted_prompt_does_not_claim_autonomous_timing():
    prompt = teacher_guidance('framework_assisted_v1')
    assert 'framework schedules' in prompt
    assert 'There is no automatic compaction or operation schedule.' not in prompt
    assert teacher_guidance('evaluation') == ''


@pytest.mark.parametrize('benchmark,launcher', [
    ('scienceworld', 'generate_scienceworld_graph_sft_deepseek_4node_idev.sh'),
    ('bcp', 'generate_ctxgraph_sft_deepseek_v4_flash_0731_9node.sh'),
])
def test_smoke_handoff_uses_fresh_assisted_collection(tmp_path, benchmark, launcher):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    script = 'smoke_assisted_graph_sft_deepseek_4node_idev.sh'
    shutil.copy(Path(__file__).resolve().parents[1] / 'scripts' / script, tmp_path / script)
    (tmp_path / launcher).write_text('set -eu\n[[ "$SFT_TEACHER_PROFILE" == framework_assisted_v1 && "$AGENT_BUDGET_MODE" == working_context && "$EXPECTED_NUM_NODES" == 4 && "$TEACHER_TP" == 4 && "$SAMPLES" == 3 && "$CANDIDATES" == 2 && -z "${RUN_DIR:-}" && -z "${ARTIFACT_ROOT:-}" && "$RUN_TAG" == 123-*-assisted-* ]]\n', encoding='utf-8')
    env = {**os.environ, 'SLURM_JOB_ID': '123', 'RUN_DIR': '/stale/run',
           'ARTIFACT_ROOT': '/stale/root', 'RUN_TAG': 'stale', 'SAMPLES': '3', 'CANDIDATES': '2'}
    subprocess.run([bash, str(tmp_path / script), benchmark], check=True, env=env, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
