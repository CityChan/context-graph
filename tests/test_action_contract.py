import asyncio
import json
from types import SimpleNamespace

import pytest

from agents.action_contract import ActionContract, PROFILE
from agents.finalizer import submit_emergency_final_answer
from agents.session_summary import iter_agent_data
from agents.utils import Agent
from tests.test_session_restart import make_agent


def encoded(tool, **arguments):
    return json.dumps({'tool': tool, 'arguments': arguments}, ensure_ascii=False)


def constrained_agent(responses, ability='LocalSearch', env=None, branch=False):
    base = make_agent(responses, enable_summary=False, graph_action_protocol='explicit_tools')
    base.config.prompt_length = 10000
    return Agent(base.llm_client, base.chat, base.tokenizer, base.config,
                 action_contract=ActionContract(ability, env, branch=branch))


@pytest.mark.parametrize('text', [
    '<function=search><parameter=query>x</parameter></function>',
    '{"tool":"search","arguments":{"query":"x"}',
    '{"tool":"search","tool":"finish","arguments":{"answer":"x"}}',
    encoded('search', query='x', unknown='field'), encoded('search', query='x', topk=True),
    encoded('open_page'), encoded('prune', node_id='n1</parameter><function=finish>'),
])
def test_invalid_or_truncated_decision_is_rejected_before_dispatch(text):
    contract = ActionContract('LocalSearch', None)
    with pytest.raises((ValueError, TypeError)):
        contract.decode(text, contract.schema())


def test_branch_and_environment_finish_restrictions():
    branch = ActionContract('LocalSearch', None, branch=True)
    for text in (encoded('branch', description='x', prompt='y'), encoded('finish', answer='x')):
        with pytest.raises(ValueError):
            branch.decode(text, branch.schema())
    branch.decode(encoded('return', message='evidence'), branch.schema())
    science = ActionContract('ScienceWorld@test', None)
    with pytest.raises(ValueError):
        science.decode(encoded('finish', message='done'), science.schema())
    with pytest.raises(ValueError):
        science.decode(encoded('action', command='look\nwait'), science.schema())


def test_discoveryworld_uses_typed_public_catalog_and_dialog_state():
    api = SimpleNamespace(isAgentInDialog=lambda index: False)
    env = SimpleNamespace(_env=api, _actions={'MOVE_DIRECTION': {'args': ['arg1']},
                                            'USE': {'args': ['arg1', 'arg2']}})
    contract = ActionContract('DiscoveryWorld@test', env)
    text = encoded('action', command={'action': 'USE', 'arg1': 3, 'arg2': 4})
    _, wire = contract.decode(text, contract.schema())
    from envs.discoveryworld_env import DiscoveryWorldEnv
    assert DiscoveryWorldEnv._command(env, wire) == {'action': 'USE', 'arg1': 3, 'arg2': 4}
    for command in ('{"action":"USE"}', {'action': 'USE', 'arg1': '3', 'arg2': 4}):
        with pytest.raises(ValueError):
            contract.decode(encoded('action', command=command), contract.schema())
    api.isAgentInDialog = lambda index: True
    with pytest.raises(ValueError):
        contract.decode(text, contract.schema())
    _, wire = contract.decode(encoded('action', command={'chosen_dialog_option_int': 1}), contract.schema())
    assert DiscoveryWorldEnv._command(env, wire) == {'chosen_dialog_option_int': 1}


def test_raw_json_policy_tokens_and_probabilities_survive_execution_and_next_input():
    async def run():
        first = encoded('search', query='evidence', topk=3)
        second = encoded('finish', answer='answer')
        agent = constrained_agent([first, second])
        wire = await agent.step()
        assert wire.startswith('<function=search>')
        assert agent.chat[-1]['content'] == first
        agent.append({'role': 'user', 'content': 'observation'})
        await agent.step()
        assert agent.chat_template_kwargs['enable_thinking'] is False
        assert all('structured_outputs' in kwargs for _, kwargs in agent.llm_client.calls)
        assert agent.llm_client.calls[1][1]['messages'][-2]['content'] == first
        outputs = [out async for _, out in iter_agent_data({'main': agent}, True, 1)]
        output = outputs[0]
        learned = [t for t, mask in zip(output['response_ids'], output['response_mask']) if mask]
        assert learned == list(map(ord, first)) + [0] + list(map(ord, second)) + [0]
        assert all(prob == -.125 for prob, mask in zip(output['response_logprobs'], output['response_mask']) if mask)
        sequence = output['prompt_ids'] + output['response_ids']
        assert all(sequence[:len(ids)] == ids for ids, _ in agent.llm_client.calls)
    asyncio.run(run())


@pytest.mark.parametrize('response,valid', [(encoded('finish', answer='result'), True),
                                         ('{"tool":"finish","arguments":', False),
                                         (encoded('search', query='more'), False)])
def test_finalizer_constrains_finish_without_fabricating_fallback(response, valid):
    async def run():
        agent = constrained_agent([response])
        env = SimpleNamespace(is_finish=False, finish=False)
        calls = []
        async def dispatch(env, response):
            calls.append(response)
            return None
        assert await submit_emergency_final_answer(agent, env, 1024, dispatch) == valid
        assert len(calls) == int(valid)
        schema = agent.llm_client.calls[0][1]['structured_outputs']['json']
        assert schema['properties']['tool']['const'] == 'finish'
        assert agent.chat[-1]['content'] == response
        assert agent.force_action_finish is False
        final_messages = agent.llm_client.calls[0][1]['messages']
        assert '<function=' not in final_messages[-1]['content']
    asyncio.run(run())


def test_swe_code_preserves_operators_and_quotes():
    contract = ActionContract('SWEVerified@test', None)
    code = 'assert 1 < 2\nprint("a & b")'
    _, wire = contract.decode(encoded('python_exec', code=code), contract.schema())
    from agents.agent_text import extract_fn_call
    assert extract_fn_call(wire)['arguments']['code'] == code


@pytest.mark.parametrize('actor,bypass', [(True, False), (False, True)])
def test_training_rejects_mixed_guided_and_raw_probability_denominators(actor, bypass):
    from omegaconf import OmegaConf
    from agents.action_contract import validate_training_contract
    config = OmegaConf.create({'actor_rollout_ref': {'actor': {'use_rollout_log_probs': actor}},
                               'algorithm': {'rollout_correction': {'bypass_mode': bypass}}})
    with pytest.raises(ValueError, match='recomputed'):
        validate_training_contract(config)


@pytest.mark.parametrize('ability', ['LocalSearch', 'SWEVerified@test', 'ScienceWorld@test'])
def test_emitted_action_schemas_are_valid_json_schema(ability):
    import jsonschema
    for branch in (False, True):
        contract = ActionContract(ability, None, branch=branch)
        schema = contract.schema()
        jsonschema.Draft202012Validator.check_schema(schema)
        decision = {'tool': 'graph_read', 'arguments': {'offset': 0}}
        jsonschema.validate(decision, schema)
        assert contract.decode(json.dumps(decision), schema)[0] == decision


@pytest.mark.parametrize('room', [0, 9, 10, 255])
def test_tiny_budget_does_not_launch_truncated_json_request(room):
    async def run():
        agent = constrained_agent([])
        assert await agent.step(max_new_tokens=room) is None
        assert agent.llm_client.calls == []
        assert agent.action_budget_skips == 1
        assert agent.action_protocol_errors == 0
    asyncio.run(run())


def test_branch_budget_protects_a_real_model_return_and_never_reports_last_search():
    async def run():
        report = encoded('return', message='Source 42: verified fact; remaining claim unresolved.')
        agent = constrained_agent([encoded('search', query='evidence'), report], branch=True)
        dispatched = []
        async def tool(response):
            dispatched.append(response)
            return 'source 42 evidence ' * 10000
        result = await agent.react(tool, max_turn=5, max_tokens=2000,
                                   should_continue=lambda r: not r.startswith('<function=return>'))
        assert result['last_response'].startswith('<function=return>')
        assert result['iteration'] == 2
        assert len(dispatched) == 1
        assert agent.forced_branch_returns == 1
        assert agent.action_protocol_errors == 0
        assert agent.response_tokens_used() <= 2000
        assert agent.llm_client.calls[-1][1]['structured_outputs']['json']['properties']['tool']['const'] == 'return'
        assert agent.chat[-1]['content'] == report
        assert all(call['max_len'] - len(ids) >= 256 for ids, call in agent.llm_client.calls)
    asyncio.run(run())


def test_explicit_search_prompt_has_one_protocol_and_preserves_question():
    from agents.prompts import create_chat
    question = 'Which book? Literal <parameter=example> in the question.'
    chat = create_chat(question, 'search_graph', explicit_graph_tools=True)
    assert chat[1]['content'] == 'Question: ' + question
    assert '<function=' not in chat[0]['content']
    assert 'unlimited thinking' not in str(chat)
    assert 'checkpoint' not in str(chat)
