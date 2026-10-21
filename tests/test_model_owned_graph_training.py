import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
from hydra import compose, initialize_config_dir

from tests.graph_training_fixtures import _setup

ROOT = Path(__file__).resolve().parents[1]


def protocol_config(tmp_path, protocol=''):
    bash = shutil.which('bash')
    if not bash:
        pytest.skip('Bash unavailable')
    script = tmp_path / 'protocol.sh'
    script.write_text('\n'.join([
        'set -eu',
        f'source "{(ROOT / "scripts/configure_graph_training_protocol.sh").as_posix()}"',
        'printf "%s\\n" "${GRAPH_PROTOCOL_ARGS[@]}"',
        'printf "%s\\n" "+actor_rollout_ref.rollout.plugin.structured_graph_controller=$BC_STRUCTURED_GRAPH_CONTROLLER" "+actor_rollout_ref.rollout.plugin.controller_owned_tool_formatting=$BC_CONTROLLER_OWNED_TOOL_FORMATTING" "+actor_rollout_ref.rollout.plugin.auto_bind_branch_edges=$BC_AUTO_BIND_BRANCH_EDGES" "+actor_rollout_ref.rollout.plugin.consolidation_interval=$CONSOLIDATION_INTERVAL" "+actor_rollout_ref.rollout.plugin.auto_prune_max_active=$AUTO_PRUNE_MAX_ACTIVE"',
    ]) + '\n', encoding='utf8', newline='\n')
    env = dict(os.environ, ADV_ESTIMATOR='graphrpo', BC_CTXGRAPH_PROTOCOL=protocol,
               CONSOLIDATION_INTERVAL='2', AUTO_PRUNE_MAX_ACTIVE='3')
    result = subprocess.run([bash, script.as_posix()], env=env, capture_output=True,
                            text=True, timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    assert result.returncode == 0, result.stderr
    with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
        return compose(config_name='ppo_trainer', overrides=[line for line in result.stdout.splitlines() if line])


def test_default_launcher_protocol_disables_scheduled_and_heuristic_edits(tmp_path):
    config = protocol_config(tmp_path)
    from agents.action_contract import validate_training_contract
    validate_training_contract(config)
    p = config.actor_rollout_ref.rollout.plugin
    assert p.graph_action_protocol == 'explicit_tools'
    assert p.action_output_profile == 'typed_action_json_v1'
    assert not p.structured_graph_controller and not p.controller_owned_tool_formatting
    assert p.consolidation_interval == p.initial_consolidation_turn == p.auto_prune_max_active == 0
    assert not p.auto_bind_branch_edges and not p.enable_retrieval_memory


@pytest.mark.parametrize('edit', [False, True])
def test_launcher_protocol_leaves_timing_to_policy_and_trains_decisions(tmp_path, monkeypatch, edit):
    p = protocol_config(tmp_path).actor_rollout_ref.rollout.plugin
    actions = [{'tool': 'search', 'arguments': {'query': 'gold'}}]
    if edit:
        actions.append({'tool': 'prune', 'arguments': {'node_id': 'n2'}})
    else:
        actions.extend([{'tool': 'search', 'arguments': {'query': 'gold'}}] * 4)
    actions.append({'tool': 'finish', 'arguments': {'answer': 'gold'}})
    responses = [json.dumps(a) for a in actions]
    module, item, context, client, *_ = _setup(monkeypatch, responses)
    context.config.actor_rollout_ref.rollout.plugin.update(p)
    output, = asyncio.run(module.process_item(item, context))
    stats = output.extra_fields['env_stats']
    assert len(client.calls) == len(responses)
    assert stats['consol_attempts'] == 0
    assert stats['graph_op_attempts'] == int(edit)
    assert output.reward_score == 1
    trained = ''.join(chr(t) for t, m in zip(output.response_ids, output.response_mask) if m)
    assert trained.replace('\0', '') == ''.join(responses)
    assert bool(any(output.extra_fields['graph_decision_mask'])) == edit
    assert all(event['source'] != 'heuristic' for event in output.extra_fields['graph_trace']['events'])


def test_controller_requires_explicit_protocol_selection(tmp_path):
    p = protocol_config(tmp_path, 'controller').actor_rollout_ref.rollout.plugin
    assert p.structured_graph_controller and p.controller_owned_tool_formatting
    assert p.consolidation_interval == 2 and p.auto_prune_max_active == 3
