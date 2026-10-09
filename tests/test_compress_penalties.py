from types import SimpleNamespace

from agents.compress_penalties import apply_compress_penalties


class Tokenizer:
    eos_token_id = 0
    def encode(self, text, **kw):
        return [ord(c) for c in text]


def agent(turns):
    tokenizer = Tokenizer()
    labels = {}
    chats = [{'role':'user', 'content':'task'}] + [{'role':'assistant','content':t} for t in turns]
    completions = [None] + [{'choices':[{'finish_reason':'stop','message':{
        'raw_output_ids': tokenizer.encode(t) + [0]}}]} for t in turns]
    return SimpleNamespace(chat=chats, chat_completions=completions, tokenizer=tokenizer,
        chat_ids=[tokenizer.encode(t['content']) for t in chats],
        set_process_reward=lambda turn, reward: labels.update({turn:reward}), labels=labels)


def call(name, parameter, value):
    return f'<function={name}><parameter={parameter}>{value}</parameter></function>'


def test_compress_penalties_cover_duplicate_actions_and_branch_returns():
    main = agent([call('search','query','alpha beta'),call('search','query','beta alpha'),
        call('open_page','docid','1'),call('open_page','docid','1'),
        call('branch','prompt','find alpha beta'),call('branch','prompt','find alpha beta'),
        call('branch','prompt','different task')])
    child = agent([call('return','message','x'*1600)])
    stats = apply_compress_penalties({'main':main,'child':child},context_limit=100000,branch_limit=2)
    assert {k:v for k,v in stats.items() if v} == {'compress_dup_search':2,'compress_dup_branch':2,'compress_ret_redund':1}
    assert main.labels == {2:-.3,4:-.3,6:-.5,7:-1.}
    assert child.labels == {1:-.3}


def test_no_eos_is_not_alone_a_truncation_and_explicit_cap_is_penalized():
    a = agent(['test', 'more'])
    for completion in a.chat_completions[1:]:
        completion['choices'][0]['message']['raw_output_ids'] = [1,2]
    a.chat_completions[2]['choices'][0]['message']['extra_data'] = {'max_tokens':2}
    stats = apply_compress_penalties({'main':a},context_limit=10000,branch_limit=2)
    assert {k:v for k,v in stats.items() if v} == {'compress_trunc':1} and a.labels == {2:-1.}


def test_redundancy_only_after_half_context_and_only_assistant_tokens():
    a = agent(['abcdefgh '*50, 'abcdefgh '*50])
    stats = apply_compress_penalties({'main':a},context_limit=600,branch_limit=2)
    assert {k:v for k,v in stats.items() if v} == {'compress_redund':1}
    assert 1 not in a.labels and a.labels[2] == -.5


def test_compress_labels_reach_real_graph_agent_training_mask(monkeypatch):
    import asyncio
    from tests.test_graph_rpo_continuation import _setup, SEARCH, finish
    module, item, context, *_ = _setup(monkeypatch,[SEARCH,SEARCH,finish('gold')])
    context.config.algorithm.graphrpo_memory_only = False
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.graph_rpo_credit_backend = 'evidence'
    plugin.graph_rpo_scope_process_reward = False
    plugin.consolidation_interval = 100
    plugin.process_reward = ['compress']
    item.non_tensor_batch['extra_info'][0]['graph_rpo_gold_docids'] = ['1']
    outputs = asyncio.run(module.process_item(item,context))
    assert outputs[0].extra_fields['env_stats']['compress_dup_search'] == 1
    assert -.3 in outputs[0].extra_fields['process_reward_mask']
