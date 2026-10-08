import asyncio
import json
import re

import pytest

from agents import supo_agent as supo
from scripts.eval_bcp_qwen38 import summarize
from tests.test_supo import setup, SUMMARY
from tests.test_agentfold import SEARCH, FINISH


def test_control_roles_override_thinking_without_changing_research(monkeypatch):
    monkeypatch.setenv('QWEN_ENABLE_THINKING', 'true')
    env, context, item, client = setup(monkeypatch, [SEARCH, SUMMARY, FINISH], observation='overflow ' * 500)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.apply_chat_template_kwargs = dict(enable_thinking=True)
    plugin.val_max_turn = 3
    renders = []
    original = context.tokenizer.apply_chat_template
    def render(messages, **kwargs):
        renders.append((messages, kwargs.copy()))
        return original(messages, **kwargs)
    monkeypatch.setattr(context.tokenizer, 'apply_chat_template', render)
    out = asyncio.run(supo.process_item(item, context))
    records = out.extra_fields['model_contexts']
    assert [r['phase'] for r in records] == ['action', 'summary', 'final']
    for record in records:
        kwargs = next(k for m, k in renders if m == record['messages'])
        assert kwargs['enable_thinking'] is (record['phase'] == 'action')
    assert 'structured_outputs' not in client.calls[0][1]
    for idx in (1, 2):
        constraint = client.calls[idx][1]['structured_outputs']
        assert constraint == records[idx]['structured_outputs']
        assert re.fullmatch(constraint['regex'], records[idx]['response'])
    assert records[1]['messages'][0]['content'] == supo.SUMMARY_SYSTEM
    assert 'Find gold.' in str(records[1]['messages'])
    assert 'Use search tools.' not in str(records[1]['messages'])
    assert 'overflow' not in str(records[1]['messages'])
    assert records[2]['messages'][0]['content'] == supo.FINAL_SYSTEM
    assert plugin.apply_chat_template_kwargs.enable_thinking is True
    assert env.actions == [SEARCH, FINISH]


def test_later_summary_sees_previous_summary_and_only_retained_history():
    messages = supo.summary_context('original task', 'known evidence [1]', [
        dict(role='assistant', content=SEARCH), dict(role='user', content='new evidence [2]')])
    assert messages[0] == dict(role='system', content=supo.SUMMARY_SYSTEM)
    assert all(s in messages[1]['content'] for s in ['original task', 'known evidence [1]', 'new evidence [2]'])


def test_bounded_control_format_rejects_loops_and_extra_tools():
    summary = supo.control_output('summary', 1024)['regex']
    assert re.fullmatch(summary, '<summary>Evidence René [1]\nStill unknown.</summary>')
    assert not re.fullmatch(summary, '<summary>' + 'The Age of Winter ' * 100 + '</summary>')
    assert not re.fullmatch(summary, '<summary>notes</summary>' + SEARCH)
    final = supo.control_output('final', 1024)['regex']
    assert re.fullmatch(final, FINISH)
    assert not re.fullmatch(final, 'Explanation: unverified story\n' + FINISH)
    assert not re.fullmatch(final, SEARCH)
    assert not re.fullmatch(final, FINISH[:-11])
    assert supo.control_output('final', 96) is None
    assert supo.control_output('action', 1024) is None


def test_summary_contract_ignoring_server_cannot_change_memory(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, '<summary>' + 'x' * 470 + '</summary>'],
                                       observation='overflow ' * 500)
    out = asyncio.run(supo.process_item(item, context))
    assert out.extra_fields['termination_reason'] == 'invalid_summary'
    assert out.extra_fields['env_stats']['summary_restarts'] == 0
    assert env.actions == [SEARCH]


def test_text_repetition_warning_does_not_redefine_token_zero_audit():
    from scripts.audit_supo_smoke import repetitive_text
    assert repetitive_text('or "The Age of Winter" or "The Age of Spring" ' * 8)
    assert not repetitive_text('Evidence [1] contradicts [2]; the answer remains unknown.')


def test_summary_failure_is_saved_and_fails_batch_summary(tmp_path):
    manifest = dict(source_sha256='hash', indices=[0, 1, 2], commit='sha', model_path='snapshot',
                    method='supo', config={}, seed=42, judge_model='judge')
    for rank in range(3):
        (tmp_path / f'manifest-{rank}.json').write_text(json.dumps(manifest))
        row = dict(source_index=rank, task_reward=0, is_finish=False, status='ok',
                   env_stats=dict(environment_steps=6, invalid_summary=1))
        (tmp_path / f'results-{rank}.jsonl').write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match='summary-format'):
        summarize(tmp_path)
    report = json.loads((tmp_path / 'summary.json').read_text())
    assert report['summary_format_failures'] == 3
    assert report['format_retry_failures'] == 0


@pytest.mark.parametrize('restarts,invalid,finish,passed', [(1, 0, True, True),
    (0, 0, True, False), (1, 1, True, False), (1, 0, False, False)])
def test_smoke_checks_summary_and_finish_independently_of_accuracy(tmp_path, restarts, invalid, finish, passed):
    from scripts.audit_supo_smoke import audit
    manifest = dict(source_sha256='hash', indices=[0, 1, 2], commit='sha', model_path='snapshot',
                    method='supo', config={}, seed=42, judge_model='judge')
    for idx in range(3):
        stats = dict(environment_steps=3, summary_restarts=restarts, invalid_summary=invalid)
        row = dict(source_index=idx, task_reward=0, is_finish=finish, status='ok', env_stats=stats)
        (tmp_path / f'manifest-{idx}.json').write_text(json.dumps(manifest))
        (tmp_path / f'results-{idx}.jsonl').write_text(json.dumps(row))
        trajectory = dict(env_stats=stats, termination_reason='finish' if finish else 'token_limit')
        (tmp_path / f'trajectory-{idx}.json').write_text(json.dumps(trajectory))
    report = audit(tmp_path)
    assert report['passed'] is passed and report['summary']['task_accuracy'] == 0
    assert (tmp_path / 'supo-smoke-audit.json').is_file()
