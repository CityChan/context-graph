import asyncio
import re

import pytest

from agents.task_action_recovery import constraint, has_task_call
from tests.test_session_restart import make_agent, Tokenizer

SEARCH = '<function=search><parameter=query>film critic published book</parameter></function>'
FINISH = '<function=finish><parameter=answer>Unknown</parameter></function>'
RETURN = '<function=return><parameter=message>No verified answer yet.</parameter></function>'


class ThinkingTokenizer(Tokenizer):
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = super().apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if add_generation_prompt:
            text += '<assistant><think>' if kwargs.get('enable_thinking', True) else '<assistant><think></think>'
        return self.encode(text) if tokenize else text


def recovering_agent(responses, role='main'):
    agent = make_agent(responses, enable_summary=False, token_budget_mode='working_context')
    agent.tokenizer = ThinkingTokenizer()
    agent.task_recovery_role = role
    return agent


def test_retry_uses_no_thinking_one_turn_preserves_tokens_and_resumes_normal_prompt():
    async def run():
        rejected = 'Let me think. ' * 50
        agent = recovering_agent([rejected, SEARCH, SEARCH])
        first = await agent.step()
        assert 'format rejected' in first and agent.model_contexts[0]['response'] == rejected
        agent.append({'role': 'user', 'content': 'No function call was detected.'})
        assert await agent.step() == SEARCH
        assert len(agent.llm_client.calls) == 2
        prompt, kwargs = agent.llm_client.calls[-1]
        assert agent.tokenizer.decode(prompt).endswith('<assistant><think></think>')
        assert kwargs['max_new_tokens'] == 512
        assert re.fullmatch(kwargs['structured_outputs']['regex'], SEARCH)
        assert agent.model_contexts[-1]['input_ids'] == prompt
        assert agent.model_contexts[-1]['output_ids'] == agent.tokenizer.encode(SEARCH) + [0]
        assert agent.tokenizer.decode(agent.chat_ids[-1]).startswith('<assistant><think></think>')
        assert not agent.task_recovery_failed
        agent.append({'role': 'user', 'content': 'Search result evidence.'})
        assert await agent.step() == SEARCH
        assert agent.tokenizer.decode(agent.llm_client.calls[-1][0]).endswith('<assistant><think>')
        assert 'structured_outputs' not in agent.llm_client.calls[-1][1]
        assert agent.total_request_tokens == sum(len(r['input_ids']) + len(r['output_ids']) for r in agent.model_contexts)
    asyncio.run(run())


def test_partial_tool_cannot_be_dispatched_or_bypass_recovery():
    partial = '<function=branch><parameter=description>find sources</parameter>'
    assert not has_task_call(partial, 'main')
    agent = recovering_agent([partial])
    assert 'format rejected' in asyncio.run(agent.step())
    assert agent.task_recovery_pending


def test_branch_constraints_keep_return_and_exclude_finish_and_branch():
    schema = constraint('branch')['regex']
    assert re.fullmatch(schema, SEARCH) and re.fullmatch(schema, RETURN)
    assert not re.fullmatch(schema, FINISH)
    assert not has_task_call(FINISH, 'branch')
    assert not re.fullmatch(schema, '<function=branch><parameter=description>x</parameter><parameter=prompt>x</parameter></function>')


@pytest.mark.parametrize('role,tool,field,prefix,old_cap', [
    ('main', 'branch', 'prompt', '<parameter=description>Identify the critic</parameter>', 384),
    ('branch', 'return', 'message', '', 512),
    ('main', 'finish', 'explanation', '<parameter=answer>Unknown</parameter>', 256),
])
def test_recovery_does_not_force_argument_closure_at_old_character_caps(role, tool, field, prefix, old_cap):
    # Complete evidence/delegations must remain expressible past the old bound.
    value = 'Keep the exact date constraints and verify them against cited sources. ' * 10
    assert len(value) > old_cap
    opening = f'<function={tool}>{prefix}<parameter={field}>'
    text = opening + value + '</parameter></function>'
    schema = constraint(role, final=tool == 'finish')['regex']
    assert re.fullmatch(schema, text)
    assert not re.fullmatch(schema, opening + value)
    assert not re.fullmatch(schema, text + SEARCH)


def test_incomplete_recovery_at_output_cap_is_preserved_but_not_dispatched():
    async def run():
        # The fake tokenizer emits one ID per character, plus EOS. Token budget
        # exhaustion must not be hidden by injecting closing XML tags.
        partial = '<function=branch><parameter=description>Find sources</parameter><parameter=prompt>'
        partial = partial + 'x' * (511 - len(partial))
        agent = recovering_agent([partial])
        agent.task_recovery_pending = True
        result = await agent.step()
        assert 'retry rejected' in result
        assert agent.task_recovery_failed
        assert agent.model_contexts[-1]['response'] == partial
        assert len(agent.model_contexts[-1]['output_ids']) == 512
        assert agent.llm_client.calls[-1][1]['max_new_tokens'] == 512
        assert len(agent.llm_client.calls) == 1
    asyncio.run(run())


@pytest.mark.parametrize('reply,expected', [(FINISH, True), ('Still thinking in a loop.', False), (SEARCH, False)])
def test_finalizer_is_constrained_and_never_wraps_failed_reasoning(reply, expected):
    from agents.finalizer import submit_emergency_final_answer
    from tests.test_emergency_finalizer import FakeEnv, fake_action_runner
    agent = recovering_agent([reply])
    env = FakeEnv()
    assert asyncio.run(submit_emergency_final_answer(agent, env, 1024, fake_action_runner)) is expected
    schema = agent.llm_client.calls[0][1]['structured_outputs']['regex']
    assert re.fullmatch(schema, FINISH) and not re.fullmatch(schema, SEARCH)
    assert env.calls == ([FINISH] if expected else [])
    assert agent.force_task_finish is False


@pytest.mark.parametrize('branch', [False, True])
def test_failed_retry_stops_real_executor_without_repeated_calls(monkeypatch, branch):
    from tests.test_scienceworld_training import setup_rollout, item
    from envs.scienceworld_env import ScienceWorldEnv
    # Simulator-backed test double, exposing the LocalSearch role contract.
    class LocalSearch(ScienceWorldEnv):
        pass
    responses = ['No tool emitted.', 'Still no tool.']
    if branch:
        responses.insert(0, '<function=branch><parameter=description>inspect</parameter><parameter=prompt>Inspect</parameter></function>')
    module, context, client = setup_rollout(monkeypatch, responses, train=False)
    monkeypatch.setattr(module, 'select_env', lambda *args: LocalSearch)
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.scienceworld_training_reward_profile = 'binary'
    plugin.inference_task_action_recovery = True
    plugin.inference_no_tool_limit = 3
    plugin.process_reward = None
    outputs = asyncio.run(module.process_item(item(), context))
    assert len(client.calls) == 2 + int(branch)
    assert all(o.extra_fields['env_stats']['hit_format_retry_limit'] == 1 for o in outputs)
    assert all(o.extra_fields['env_stats']['task_action_recovery_requests'] == 1 for o in outputs)
    assert all(o.extra_fields['termination_reason'] == 'tool_format_retry_limit' for o in outputs)


def test_recovery_audit_requires_real_tools_and_actual_retry(tmp_path):
    import json
    from scripts.audit_graph_retry_ab import audit_task_recovery
    manifest = dict(source_sha256='hash', indices=[28, 70], commit='sha', model_path='snapshot',
                    method='contextgraph', seed=42, judge_model='judge',
                    config={'actor_rollout_ref': {'rollout': {'plugin': {'inference_task_action_recovery': True}}}})
    for rank in range(3):
        (tmp_path / f'manifest-{rank}.json').write_text(json.dumps(manifest))
        rows = []
        for index in manifest['indices'][rank::3]:
            rows.append(dict(source_index=index, task_reward=0, is_finish=False, status='ok',
                             env_stats=dict(search=1, open_page=0, task_format_retry_requests=1, task_format_recovery_successes=1)))
            (tmp_path / f'trajectory-{index}.json').write_text('{}')
        (tmp_path / f'results-{rank}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert audit_task_recovery(tmp_path)['passed']
    path = tmp_path / 'results-1.jsonl'
    row = json.loads(path.read_text())
    row['env_stats']['search'] = 0
    path.write_text(json.dumps(row) + '\n')
    assert not audit_task_recovery(tmp_path)['passed']
