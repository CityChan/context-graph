import asyncio
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('branch', [False, True])
def test_inference_no_tool_guard_covers_main_and_branch(monkeypatch, branch):
    from tests.test_scienceworld_training import setup_rollout, item
    responses = ['No complete tool call.'] * 3
    if branch:
        responses.insert(0, '<function=branch><parameter=description>inspect</parameter><parameter=prompt>Inspect the room</parameter></function>')
    module, context, client = setup_rollout(monkeypatch, responses, train=False)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.scienceworld_training_reward_profile = 'binary'
    plugin.inference_no_tool_limit = 3
    plugin.process_reward = None
    outputs = asyncio.run(module.process_item(item(), context))
    assert len(client.calls) == 3 + int(branch)
    assert all(o.extra_fields['termination_reason'] == 'no_tool_limit' for o in outputs)
    assert all(o.extra_fields['env_stats']['hit_no_tool_limit'] == 1 for o in outputs)
    assert all(not o.extra_fields['env_stats']['finalizer_attempted'] for o in outputs)


def test_inference_no_tool_guard_rejects_training(monkeypatch):
    from tests.test_scienceworld_training import setup_rollout, item
    module, context, _ = setup_rollout(monkeypatch, [], train=True)
    context.config.actor_rollout_ref.rollout.plugin.inference_no_tool_limit = 3
    with pytest.raises(ValueError, match='inference-only'):
        asyncio.run(module.process_item(item(), context))


def test_smoke_limits_are_explicit_and_default_eval_is_unchanged(monkeypatch):
    from scripts.eval_bcp_qwen38 import config_for
    args = SimpleNamespace(method='contextgraph', memory_mode='repaired', graph_action_protocol='controller')
    keys = ('EVAL_MAX_TURN', 'EVAL_SESSION_TIMEOUT', 'EVAL_EPISODE_TIMEOUT', 'EVAL_NO_TOOL_LIMIT')
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    plugin = config_for(args).actor_rollout_ref.rollout.plugin
    assert (plugin.max_turn, plugin.session_timeout, plugin.eval_episode_timeout, plugin.inference_no_tool_limit) == (100, 3600, 0, 0)
    for key, value in zip(keys, (12, 600, 900, 3)):
        monkeypatch.setenv(key, str(value))
    plugin = config_for(args).actor_rollout_ref.rollout.plugin
    assert (plugin.max_turn, plugin.val_max_turn, plugin.session_timeout, plugin.eval_episode_timeout, plugin.inference_no_tool_limit) == (12, 12, 600, 900, 3)
    args.graph_action_protocol = 'explicit_tools'
    with pytest.raises(ValueError, match='controller'):
        config_for(args)


def test_saved_response_diagnostic_needs_no_tokenizer(tmp_path):
    from scripts.diagnose_bcp_stalled_actions import diagnose
    arm = tmp_path / 'baseline'
    arm.mkdir()
    trajectory = {'env_stats': {'search': 0, 'open_page': 0},
                  'model_contexts': [{'response': '<think>unfinished', 'output_ids': [1, 2],
                                      'messages': [{'role': 'user', 'content': 'No function call was detected.'}]}]}
    (arm / 'trajectory-28.json').write_text(json.dumps(trajectory))
    (arm / 'requests-28.jsonl').write_text(json.dumps({'finish_reason': 'length', 'output_ids': [1, 2], 'max_tokens': 2}) + '\n')
    result = diagnose(tmp_path, 28)
    assert result['at_output_limit'] == 1
    assert result['finish_reasons'] == {'length': 1}
    assert result['samples'][0]['complete_xml_call_names'] == []
    assert 'No function' in result['samples'][0]['preceding_user_tail']


def test_smoke_summary_fails_zero_search_even_when_process_ok(tmp_path):
    from scripts.eval_bcp_qwen38 import summarize
    manifest = dict(source_sha256='hash', indices=[0, 1, 2], commit='sha', model_path='snapshot',
                    method='contextgraph', seed=42, judge_model='judge',
                    config={'actor_rollout_ref': {'rollout': {'plugin': {'eval_require_task_tools': True}}}})
    for rank in range(3):
        (tmp_path / f'manifest-{rank}.json').write_text(json.dumps(manifest))
        row = dict(source_index=rank, task_reward=0, is_finish=False, status='ok',
                   env_stats={'search': 1 if rank == 0 else 0, 'open_page': 0})
        (tmp_path / f'results-{rank}.jsonl').write_text(json.dumps(row) + '\n')
    with pytest.raises(RuntimeError, match='zero-task-tool'):
        summarize(tmp_path)
    assert json.loads((tmp_path / 'summary.json').read_text())['zero_tool_tasks'] == 2
