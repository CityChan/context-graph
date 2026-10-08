"""Regression for Vista's zero-tool, repeated batched-search failure."""
import asyncio
import json

import pytest

from agents import agentfold_agent as af, supo_agent as supo, memory_baseline_agent as memory
from agents import prompts
from agents.single_tool_protocol import retry_context
from agents.tool_spec import PARALLEL_TOOL_PROMPT, convert_tools_to_description, search_tool
from tests.test_agentfold import SEARCH, FINISH, compress, setup as setup_af
from tests.test_supo import setup as setup_supo
from tests.test_graph_memory_baselines import setup as setup_memory, PATCH, ANALYSIS
from tests.test_session_restart import Tokenizer


BAD = '<think>Search three leads.</think>\n' + SEARCH * 3 + '</function>'


def setup(monkeypatch, method, responses, real_prompt=False):
    if method == 'agentfold':
        runner, values = af, setup_af(monkeypatch, responses, observation='evidence [1]')
    elif method == 'supo':
        runner, values = supo, setup_supo(monkeypatch, responses, observation='evidence [1]')
    else:
        runner, values = memory, setup_memory(monkeypatch, method, responses, observation='evidence [1]')
    env, context, item, client = values
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.supo_context_threshold = 30000
    plugin.memobrain_context_threshold = 30000
    if real_prompt:
        monkeypatch.setattr(runner, 'create_chat', prompts.create_chat)
    return runner, env, context, item, client


def test_single_prompt_removes_batch_example_and_preserves_original_protocol():
    question = 'Literal task: * Or you can search multiple queries {unchanged}'
    single = prompts.create_chat(question, 'search_single')
    assert question in single[1]['content']
    assert 'Choose exactly one tool call per response' in single[1]['content']
    assert 'including multiple <function=search> actions' not in str(single)
    original = prompts.create_chat(question, 'search')
    assert original == [
        dict(role='system', content=prompts.SEARCH_SYSTEM_PROMPT + '\n\n' +
             PARALLEL_TOOL_PROMPT.format(description=convert_tools_to_description(search_tool()))),
        dict(role='user', content=prompts.SEARCH_EXAMPLE + '\n\n' +
             prompts.SEARCH_USER_PROMPT.format(Question=question))]


@pytest.mark.parametrize('method', ['agentfold', 'supo', 'memobrain', 'amem'])
def test_real_adapter_prompt_and_recovery_execute_search(monkeypatch, method):
    responses = [BAD, SEARCH]
    if method == 'memobrain':
        responses.append(json.dumps(PATCH))
    elif method == 'amem':
        responses.append(json.dumps(ANALYSIS))
    responses.append(FINISH)
    runner, env, context, item, client = setup(monkeypatch, method, responses, real_prompt=True)
    out = asyncio.run(runner.process_item(item, context))
    assert env.actions == [SEARCH, FINISH] and env.closed
    assert out.reward_score == 1
    first = str(client.calls[0][1]['messages'])
    assert 'including multiple <function=search> actions' not in first
    assert 'Choose exactly one tool call per response' in first
    retry = client.calls[1][1]['messages']
    assert retry[-2] == dict(role='assistant', content=BAD)
    assert 'no tool was executed' in retry[-1]['content']
    stats = out.extra_fields['env_stats']
    assert stats['invalid_tool'] == 1 and stats['hit_format_retry_limit'] == 0


@pytest.mark.parametrize('method', ['agentfold', 'supo', 'memobrain', 'amem'])
def test_repeated_batch_failure_stops_without_dispatch_or_memory_mutation(monkeypatch, method):
    runner, env, context, item, client = setup(monkeypatch, method, [BAD] * 10)
    out = asyncio.run(runner.process_item(item, context))
    assert len(client.calls) == 3 and env.actions == [] and env.closed
    assert out.extra_fields['termination_reason'] == 'invalid_tool_limit'
    stats = out.extra_fields['env_stats']
    assert stats['invalid_tool'] == 3 and stats['hit_format_retry_limit'] == 1
    assert stats['environment_steps'] == stats['hit_token_limit'] == 0
    assert stats.get('agentfold_folds', 0) == stats.get('memory_updates', 0) == 0


def test_valid_action_resets_consecutive_failure_limit_and_fold_still_works(monkeypatch):
    responses = [BAD, BAD, SEARCH, BAD, BAD, compress(1, 1) + SEARCH, FINISH]
    runner, env, context, item, client = setup(monkeypatch, 'agentfold', responses)
    out = asyncio.run(runner.process_item(item, context))
    assert env.actions == [SEARCH, SEARCH, FINISH]
    assert out.extra_fields['env_stats']['agentfold_folds'] == 1
    assert out.extra_fields['env_stats']['invalid_tool'] == 4
    assert out.extra_fields['termination_reason'] == 'finish'


def test_retry_excerpt_is_bounded_and_contains_rejected_tail():
    rejected, correction = retry_context('x' * 2000 + BAD, 'bad call', Tokenizer())
    assert rejected.endswith(BAD) and len(rejected) < 600
    assert 'one complete function call' in correction


def test_summary_saves_format_failure_and_rejects_clean_score(tmp_path):
    from scripts.eval_bcp_qwen38 import summarize
    manifest = dict(source_sha256='hash', indices=[0, 1, 2], commit='sha', model_path='snapshot',
                    method='agentfold', config={}, seed=42, judge_model='judge')
    for rank in range(3):
        (tmp_path / f'manifest-{rank}.json').write_text(json.dumps(manifest))
        row = dict(source_index=rank, task_reward=0, is_finish=False, status='ok',
                   env_stats=dict(environment_steps=0, hit_format_retry_limit=1))
        (tmp_path / f'results-{rank}.jsonl').write_text(json.dumps(row) + '\n')
    with pytest.raises(RuntimeError, match='tool-format retry'):
        summarize(tmp_path)
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['format_retry_failures'] == summary['zero_tool_tasks'] == 3
