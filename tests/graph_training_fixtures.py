import copy
import json
from types import SimpleNamespace
import numpy as np

def _setup(monkeypatch, responses, *, train=True):
    from agents import graph_agent_isolated
    from envs.local_search import AsyncSearchClient
    from scripts.eval_discoverybench_qwen35 import config_for
    from tests.test_session_restart import Tokenizer, Client

    calls, closed = [], []
    async def post(self, path, payload):
        calls.append((path, copy.deepcopy(payload)))
        return [{'docid': '1', 'url': 'local/1', 'title': 'Evidence', 'text': 'gold evidence'}]
    original_close = AsyncSearchClient.close
    async def close(self):
        if self not in closed:
            closed.append(self)
        await original_close(self)
    monkeypatch.setattr(AsyncSearchClient, '_post', post)
    monkeypatch.setattr(AsyncSearchClient, 'close', close)
    monkeypatch.setenv('LOCAL_SEARCH_URL', 'http://localhost:9999')
    monkeypatch.setattr(graph_agent_isolated, 'create_chat', lambda *a, **kw: [
        {'role': 'system', 'content': 'Use tools and maintain the graph when requested.'},
        {'role': 'user', 'content': 'Find gold.'},
    ])
    config = config_for('contextgraph', 32768)
    config.algorithm.adv_estimator = 'graphrpo'
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.graph_rpo_scope_process_reward = False
    plugin.workflow = 'search_graph'
    plugin.graph_rpo_credit_backend = 'outcome_only'
    plugin.graph_controller_temperature = 0.8
    plugin.controller_allow_pass = True
    plugin.consolidation_interval = 2
    plugin.max_turn = plugin.val_max_turn = 8
    plugin.session_timeout = 60
    plugin.max_traj = 10
    item = SimpleNamespace(non_tensor_batch={
        'ability': np.array(['LocalSearch']), 'uid': 'question', 'gen_uid': 'source',
        'extra_info': np.array([{'query': 'Find gold.', 'answer': 'gold',
                                'reward_mode': 'searchr1_em', 'workflow': 'search_graph'}], dtype=object),
    })
    client = Client(responses)
    context = SimpleNamespace(config=config, is_train=train, global_step=3,
                              tokenizer=Tokenizer(), llm_client=client)
    return graph_agent_isolated, item, context, client, calls, closed


SEARCH = '<function=search><parameter=query>gold</parameter></function>'
OPEN = '<function=open_page><parameter=docid>1</parameter></function>'
def finish(answer):
    return f'<function=finish><parameter=answer>{answer}</parameter></function>'
def decision(action, indices):
    return json.dumps(dict(action=action, candidate_indices=indices, summary='', relation='semantic'))
