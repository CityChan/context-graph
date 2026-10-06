import asyncio
import copy
from types import SimpleNamespace

from agents.utils import Agent
from agents.session_summary import iter_agent_data
from agents.finalizer import remaining_generation_tokens


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))

    def __call__(self, text, **kwargs):
        return {'input_ids': self.encode(text)}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = ''.join(f"<{m['role']}>{m['content']}\0" for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return self.encode(text) if tokenize else text


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def create_completion(self, ids, **kwargs):
        self.calls.append((list(ids), copy.deepcopy(kwargs)))
        content = next(self.responses)
        tokens = list(map(ord, content)) + [0]
        return {'choices': [{'message': {'content': content, 'raw_output_ids': tokens,
                                        'response_log_probs': [-0.125] * len(tokens)}}]}


def make_agent(responses, **settings):
    plugin = SimpleNamespace(enable_summary=True, summary_context_threshold=500,
                             summary_max_tokens=160, summary_max_restarts=3,
                             capture_model_contexts=True)
    for key, value in settings.items():
        setattr(plugin, key, value)
    config = SimpleNamespace(prompt_length=500, response_length=6000, plugin=plugin)
    return Agent(Client(responses), [{'role': 'system', 'content': 'rules'},
                                    {'role': 'user', 'content': 'fix the bug'}],
                 Tokenizer(), config)


def test_restart_changes_actual_model_input_preserves_budget_and_training_alignment():
    async def run():
        agent = make_agent(['inspect', '<summary>edited x.py; test next</summary>', 'finish'])
        await agent.step()
        agent.set_process_reward(len(agent.chat) - 1, -0.2)
        agent.additional_info[-1]['graph_edit_credit'] = 0.7
        agent.append({'role': 'user', 'content': 'obsolete details ' * 100})
        old_ids = copy.deepcopy(agent.chat_ids)
        used = agent.response_tokens_used()
        assert await agent.maybe_restart_session(100)
        start = agent.summary_sessions[0]['start']
        # Graph node turn references still identify exactly the same turns.
        assert agent.chat_ids[:len(old_ids)] == old_ids
        assert agent.response_tokens_used() > used
        assert len(agent.context()) < 500
        assert remaining_generation_tokens(agent) == 6000 - agent.response_tokens_used() - len(agent.get_generation_prompt())
        await agent.step()
        ids, call = agent.llm_client.calls[-1]
        assert 'obsolete details' not in str(call['messages'])
        assert 'edited x.py' in str(call['messages'])
        assert ids == agent.model_contexts[-1]['input_ids']
        assert call['max_len'] - len(ids) <= 6000 - used
        segments = [data async for _, data in iter_agent_data({'main': agent}, True, 1)]
        assert len(segments) == 2  # max_traj must not discard restarted sessions
        assert start + 1 in segments[1]['response_turn_token_indices']
        all_policy = []
        for data in segments:
            assert len(data['response_ids']) == len(data['response_mask']) == len(data['response_logprobs'])
            all_policy.extend(token for token, mask in zip(data['response_ids'], data['response_mask']) if mask)
            assert all(lp == -0.125 for lp, mask in zip(data['response_logprobs'], data['response_mask']) if mask)
        expected = [token for turn, mask in zip(agent.chat_ids, agent.token_mask)
                    for token, keep in zip(turn, mask) if keep]
        assert all_policy == expected  # no lost or duplicated policy tokens
        from agents.trajectory_capture import serialize_agent_trajectories
        audit = serialize_agent_trajectories({'main': agent})
        assert len(audit) == 2
        assert audit[1]['summary_session_index'] == 1
        assert 'obsolete details' not in str(audit[1]['messages'])
        assert 'edited x.py' in str(audit[1]['messages'])
        # Each sampled prefix is an exact prefix of its exported training segment.
        offset = 0
        for data in segments:
            sequence = data['prompt_ids'] + data['response_ids']
            segment_calls = agent.llm_client.calls[:2] if offset == 0 else agent.llm_client.calls[2:]
            for input_ids, _ in segment_calls:
                assert sequence[:len(input_ids)] == input_ids
            offset += 1
    asyncio.run(run())


def test_graph_credit_and_process_rewards_keep_absolute_turn_mapping_after_restart():
    async def run():
        agent = make_agent(['<summary>checkpoint</summary>', 'edit graph'])
        agent.append({'role': 'user', 'content': 'evidence ' * 100})
        assert await agent.maybe_restart_session()
        await agent.step()
        turn = len(agent.chat) - 1
        agent.set_process_reward(turn, -0.2)
        agent.additional_info[turn]['graph_edit_credit'] = 0.7
        segments = [x async for x in agent.session_data()]
        assert turn not in segments[0]['response_turn_token_indices']
        indices = segments[1]['response_turn_token_indices'][turn]
        assert indices
        assert all(segments[1]['process_reward_mask'][i] == -0.2 for i in indices)
        assert all(segments[1]['graph_edit_credit_mask'][i] == 0.7 for i in indices)
        assert all(segments[1]['graph_decision_mask'][i] == 1 for i in indices)
        assert sum(segments[0]['graph_decision_mask']) == 0
    asyncio.run(run())


def test_large_summary_does_not_drop_original_task_and_restart_limit_is_enforced():
    async def run():
        agent = make_agent(['<summary>' + 'x' * 500 + '</summary>'])
        agent.append({'role': 'user', 'content': 'evidence ' * 100})
        original = copy.deepcopy(agent.chat[:2])
        assert not await agent.maybe_restart_session()
        assert agent.chat[:2] == original
        limited = make_agent(['<summary>done</summary>'], summary_max_restarts=1)
        limited.append({'role': 'user', 'content': 'evidence ' * 100})
        assert await limited.maybe_restart_session()
        limited.append({'role': 'user', 'content': 'more evidence ' * 100})
        assert not await limited.maybe_restart_session()
        assert len(limited.llm_client.calls) == 1
    asyncio.run(run())


def test_multiple_restarts_keep_global_indices_and_never_reset_budget():
    async def run():
        agent = make_agent(['<summary>first</summary>', 'action', '<summary>second</summary>', 'done'])
        for n in range(2):
            agent.append({'role': 'user', 'content': f'observation {n} ' * 70})
            before = remaining_generation_tokens(agent)
            assert await agent.maybe_restart_session()
            assert remaining_generation_tokens(agent) < before
            await agent.step()
        data = [x async for x in agent.session_data()]
        assert len(data) == 3
        mapped = [turn for x in data for turn in x['response_turn_token_indices']]
        assert len(mapped) == len(set(mapped))
        assert agent.summary_sessions[-1]['start'] + 1 in mapped
    asyncio.run(run())


def test_invalid_summary_is_not_installed_and_cost_is_not_refunded():
    async def run():
        agent = make_agent(['<function=finish>oops</function>'])
        agent.append({'role': 'user', 'content': 'evidence ' * 100})
        before = remaining_generation_tokens(agent)
        assert not await agent.maybe_restart_session()
        assert not getattr(agent, 'summary_sessions', [])
        assert remaining_generation_tokens(agent) < before
        assert not await agent.maybe_restart_session()  # retry cooldown
    asyncio.run(run())


def test_exhausted_budget_cannot_be_replenished_by_summary():
    async def run():
        agent = make_agent([])
        agent.append({'role': 'user', 'content': 'x' * 5980})
        before = copy.deepcopy(agent.chat)
        assert not await agent.maybe_restart_session(100)
        assert agent.chat == before
        assert await agent.step() is None
        assert not agent.llm_client.calls
    asyncio.run(run())
