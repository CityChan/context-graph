import asyncio
import json

import pytest

from agents import agentfold_agent as af, prompts
from tests.test_agentfold import setup, SEARCH, OPEN, FINISH, compress


def test_upstream_singleton_and_consecutive_ranges_preserve_whole_blocks():
    steps = [af.Step(1, 1, 'one')]
    action, replacement = af.parse_response(OPEN + '<compress>{"compress_range":[1],"compress_text":"source [1]"}</compress>', steps)
    assert action == OPEN and replacement == [af.Step(1, 1, 'source [1]', True)]
    steps = [af.Step(1, 2, 'old', True), af.Step(3, 3, 'new')]
    folded = af.fold_suffix(steps, dict(compress_range=[1, 2, 3], compress_text='all'))
    assert folded == [af.Step(1, 3, 'all', True)]
    with pytest.raises(ValueError, match='whole blocks'):
        af.fold_suffix(steps, dict(compress_range=[2, 3], compress_text='split'))
    with pytest.raises(ValueError, match='consecutive'):
        af.fold_suffix(steps, dict(compress_range=[1, 1, 3], compress_text='gap'))


def test_missing_fold_recovers_from_live_contract_without_repeating_tool(monkeypatch):
    env, context, item, client = setup(monkeypatch, [SEARCH, OPEN,
        '<compress>{"compress_range":[1],"compress_text":"gold evidence [1]"}</compress>' + OPEN,
        compress(1, 2) + SEARCH, FINISH], observation='ACTUAL_EVIDENCE')
    out = asyncio.run(af.process_item(item, context))
    assert env.actions == [SEARCH, OPEN, SEARCH, FINISH]
    retry = client.calls[2][1]['messages'][-1]['content']
    assert 'end ID must be 1' in retry and 'Valid start IDs: [1]' in retry
    assert '"compress_range": [1, 1]' in retry
    after = client.calls[3][1]['messages'][-1]['content']
    assert 'end ID must be 2' in after and 'Valid start IDs: [1,2]' in after
    assert '[Compressed Step 1]' in after
    stats = out.extra_fields['env_stats']
    assert stats['agentfold_folds'] == 2 and stats['invalid_tool'] == 1
    assert stats['main_len'] == sum(stats[k] for k in
        ('generated_tokens', 'observation_tokens', 'instruction_tokens', 'feedback_tokens'))


def test_raw_reasoning_is_audited_but_not_replayed_as_evidence(monkeypatch):
    text = '<think>IMAGINED_RESULT</think>' + SEARCH
    env, context, item, client = setup(monkeypatch, [text, FINISH], observation='REAL_RESULT')
    out = asyncio.run(af.process_item(item, context))
    subsequent = str(client.calls[1][1]['messages'])
    assert 'IMAGINED_RESULT' not in subsequent and 'REAL_RESULT' in subsequent
    assert out.extra_fields['model_contexts'][0]['response'] == text


def test_dedicated_prompt_preserves_task_and_excludes_incompatible_demo():
    task = 'Verbatim question {with braces}'
    messages = prompts.create_chat(task, 'search_agentfold')
    assert task in messages[1]['content']
    assert prompts.SEARCH_EXAMPLE not in messages[1]['content']
    assert 'Only call one function at a time' in messages[0]['content']
    assert prompts.SEARCH_EXAMPLE in prompts.create_chat(task, 'search_single')[1]['content']


def test_incomplete_call_at_output_limit_cannot_dispatch(monkeypatch):
    text = '<function=open_page><parameter=docid>'
    env, context, item, client = setup(monkeypatch, [text] * 3)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.turn_max_new_tokens = len(text) + 1
    out = asyncio.run(af.process_item(item, context))
    assert not env.actions
    assert out.extra_fields['env_stats']['format_errors_at_output_limit'] == 3


@pytest.mark.parametrize('stop,folds,passed', [('finish', 1, True), ('finish', 0, False),
                                            ('invalid_tool_limit', 1, False)])
def test_smoke_checks_folding_and_format_failure_without_accuracy_gate(tmp_path, stop, folds, passed):
    from scripts.audit_agentfold_smoke import audit
    manifest = dict(source_sha256='hash', indices=[0, 1, 2], commit='sha', model_path='snapshot',
                    method='agentfold', config={}, seed=42, judge_model='judge')
    for idx in range(3):
        stats = dict(environment_steps=2, agentfold_folds=folds)
        row = dict(source_index=idx, task_reward=0, is_finish=stop == 'finish', status='ok', env_stats=stats)
        (tmp_path / f'manifest-{idx}.json').write_text(json.dumps(manifest))
        (tmp_path / f'results-{idx}.jsonl').write_text(json.dumps(row))
        (tmp_path / f'trajectory-{idx}.json').write_text(json.dumps(dict(env_stats=stats, termination_reason=stop)))
    result = audit(tmp_path)
    assert result['passed'] is passed
    assert result['summary']['task_accuracy'] == 0
    assert (tmp_path / 'agentfold-smoke-audit.json').is_file()
