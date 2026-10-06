import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents.evidence_credit import assign_evidence_credits, covered_documents, expanded_documents, gold_docids


SOURCE = "Roger Ebert reviewed movies for the Chicago Sun Times for forty six years."


class Sink:
    def __init__(self): self.credits = {}
    def add_graph_edit_credit(self, turn, credit): self.credits[turn] = credit


def event(op, turn, **kwargs):
    return dict(source="model", success=True, op=op, assistant_turn_index=turn,
                evidence_document_count=1, args={}, **kwargs)


def score(events, documents=None):
    sink = Sink()
    metrics = assign_evidence_credits(agent=sink, graph_trace={"events": events},
        gold={"17"}, documents=documents or [{"docid": "17", "text": SOURCE}], plugin_config={})
    return sink.credits, metrics


def test_branch_first_discovery_duplicate_and_refined_retry():
    events = []
    for turn, (retrieved, seen) in enumerate([(["17"], []), (["17"], ["17"]), (["18"], ["17"])], 1):
        e = event("branch", turn, evidence_seen_before=seen, evidence_retrieved=retrieved)
        e["args"] = {"description": "Find critic", "prompt": "Search film critic review"}
        events.append(e)
    credit, metrics = score(events)
    assert credit == {1: 1.0, 2: -0.2, 3: 0.0}
    assert metrics["graph_rpo_duplicate_branches"] == 1
    assert all(e["graph_rpo_outcome_gated"] is False for e in events)
    events[1]['evidence_expanded_documents'] = ['17']
    assert score(events)[0][2] == 0


def test_opening_more_of_same_document_is_not_duplicate():
    before = [{'docid': '17', 'text': SOURCE}]
    assert expanded_documents(before, before) == set()
    assert expanded_documents(before, [{'docid': '17', 'text': SOURCE + ' He retired in 2013.'}]) == {'17'}
    repeated = '(This page was already seen in a previous search. Here, a shorter snippet is shown.) ' + SOURCE
    assert expanded_documents(before, [{'docid': '17', 'text': repeated}]) == set()


def test_view_loss_and_recovery_not_archive_size_or_ids():
    events = [event("merge", 1, credit_view_before=SOURCE, credit_view_after="docid: 17"),
              event("select", 2, credit_view_before="docid: 17", credit_view_after=SOURCE),
              event("prune", 3, credit_view_before=SOURCE, credit_view_after=SOURCE),
              event("pass", 4)]
    credit, metrics = score(events)
    assert credit == {1: -1.0, 2: 1.0, 3: 0.0, 4: 0.0}
    assert metrics["graph_rpo_evidence_lost"] == 1
    assert covered_documents(SOURCE, [], {"17"}) == set()
    events[0]["evidence_document_count"] = 0
    assert score([events[0]])[0][1] == 0


@pytest.mark.parametrize("labels", [None, [], "17", [True], [None], [""]])
def test_labels_fail_closed(labels):
    with pytest.raises(ValueError): gold_docids({"graph_rpo_gold_docids": labels})


def test_preparation_does_not_change_prompt_or_source():
    from scripts.prepare_graph_evidence_data import attach
    rows = [{"prompt": [{"role": "user", "content": "question"}],
             "extra_info": {"query": "question", "answer": "answer"}}]
    original = copy.deepcopy(rows)
    result = attach(rows, {"question": [17, "17"]})
    assert rows == original
    assert result[0]["prompt"] == original[0]["prompt"]
    assert result[0]["extra_info"]["graph_rpo_gold_docids"] == ["17"]
    with pytest.raises(ValueError): attach(rows, {})


def test_local_credit_not_diluted_by_main_or_branch_length():
    import torch
    from verl.trainer.ppo.core_algos import compute_graphrpo_advantage, compute_graphrpo_loss_weights
    for width in [4, 1000]:
        mask = torch.ones((3, width))  # two streams in episode a, one in b
        credit = torch.zeros_like(mask)
        decision = torch.zeros_like(mask)
        credit[0, 0] = -0.2
        decision[0, :2] = 1  # include a zero-reward decision
        uid = np.array(["q"] * 3)
        gen = np.array(["a", "a", "b"])
        adv, _ = compute_graphrpo_advantage(torch.zeros_like(mask), mask, uid, gen,
            graph_edit_credit_mask=credit, graph_decision_mask=decision,
            config={"graphrpo_alpha": 0.5, "graphrpo_normalize_decision_tokens": True})
        weights = compute_graphrpo_loss_weights(mask, uid, gen)
        assert float((adv * weights).sum()) == pytest.approx(-0.025)
        assert adv[1:].abs().sum() == 0
    with pytest.raises(ValueError, match="requires"):
        compute_graphrpo_advantage(torch.zeros_like(mask), mask, uid, gen,
            config={"graphrpo_normalize_decision_tokens": True})


@pytest.mark.parametrize('is_train', [True, False])
def test_training_rollout_routes_branch_credit_and_keeps_oracle_out_of_prompts(monkeypatch, is_train):
    from verl import DataProto
    import agents.graph_agent_isolated as module
    from scripts.eval_agent_benchmarks import config_for
    from tests.test_session_restart import Client, Tokenizer

    class Env:
        def __init__(self, *args):
            self.stats = {}
            self.evidence_documents = []
            self.is_finish = False
            self.env_fail = False
        async def init_env(self, item):
            self.instance_info = {"problem_statement": "Find the critic"}
        async def run_action(self, response):
            if "<function=finish>" in response:
                self.is_finish = True
                return {"observation": "Done"}
            self.evidence_documents.append({"docid": "SECRET_GOLD_ID", "text": SOURCE})
            return {"observation": SOURCE}
        async def get_reward(self): return ("wrong", 0.0)
        async def close(self): pass

    monkeypatch.setattr(module, "select_env", lambda *args: Env)
    config = config_for("scienceworld", "contextgraph", 65536)
    config.algorithm.adv_estimator = "graphrpo"
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.graph_rpo_credit_backend = "evidence"
    plugin.graph_rpo_scope_process_reward = False
    plugin.graph_branch_history = True
    plugin.consolidation_interval = 1
    plugin.graph_controller_temperature = 0.8
    plugin.max_turn = 12
    plugin.final_answer_reserve = 0
    config.actor_rollout_ref.rollout.response_length = 100000
    responses = [
        '<function=branch><parameter=description>Find critic</parameter><parameter=prompt>Find critic</parameter></function>',
        '<function=search><parameter=query>film critic</parameter></function>',
        '<function=return><parameter=message>' + SOURCE + '</parameter></function>',
        '<function=finish><parameter=answer>Wrong</parameter></function>',
    ]
    task = DataProto()
    task.non_tensor_batch = {"ability": np.array(["BrowseCompPlus"], dtype=object),
        "extra_info": np.array([{"query": "Find the critic", "answer": "X", "workflow": "search_graph",
            "graph_rpo_gold_docids": ["SECRET_GOLD_ID"]}], dtype=object), "uid": np.array(["q"], dtype=object)}
    task.meta_info = {"generation_kwargs": {}}
    class ControllerClient(Client):
        async def create_completion(self, ids, **kwargs):
            if kwargs.get('structured_outputs'):
                assert kwargs['sampling_params']['temperature'] == (0.8 if is_train else 0)
                sub = Client([json.dumps({'action': 'pass', 'candidate_indices': [], 'summary': '', 'relation': 'causal'})])
                result = await sub.create_completion(ids, **kwargs)
                self.calls.extend(sub.calls)
                return result
            return await super().create_completion(ids, **kwargs)
    client = ControllerClient(responses)
    if not is_train:
        del task.non_tensor_batch['extra_info'][0]['graph_rpo_gold_docids']
    outputs = asyncio.run(module.process_item(task, SimpleNamespace(config=config, tokenizer=Tokenizer(),
        llm_client=client, is_train=is_train, global_step=0)))
    assert outputs
    fields = [out.extra_fields for out in outputs]
    assert any(max(row["graph_edit_credit_mask"]) > 0 for row in fields) == is_train
    assert all("SECRET_GOLD_ID" not in ''.join(map(chr, ids)) for ids, _ in client.calls)
    assert "[Previous branch attempts]" in ''.join(map(chr, client.calls[-1][0]))
    branch = next(e for e in fields[0]["graph_trace"]["events"] if e['op'] == 'branch')
    if is_train:
        assert branch['graph_rpo_delta'] == 1.0
        passes = [e for e in fields[0]['graph_trace']['events'] if e['op'] == 'pass']
        assert passes and passes[0]['graph_rpo_delta'] == 0.0
        assert fields[0]['env_stats']['graph_rpo_decision_tokens'] > 0
    else:
        assert 'graph_rpo_delta' not in branch
    assert branch['assistant_turn_index'] > 0


def test_audit_requires_backend_and_deduplicates_episode_streams(tmp_path):
    from scripts.audit_evidence_graph_credit import audit
    e = event("pass", 1)
    score([e])
    row = {"gen_uid": "a", "graph_trace": {"final_hash": "x", "events": [e]}}
    path = tmp_path / "rollout.jsonl"
    path.write_text(json.dumps(row) + '\n' + json.dumps(row), encoding="utf8")
    assert audit([path])["decisions"] == 1
    row['graph_trace']['events'][0]['graph_rpo_delta'] = 0.5
    path.write_text(json.dumps(row), encoding='utf8')
    with pytest.raises(ValueError, match='components'): audit([path])


def test_retrieval_records_only_displayed_documents_and_rejects_failed_open():
    from collections import Counter
    from envs.local_search import LocalSearch
    class Search:
        async def search(self, *args):
            return [{'docid': str(i), 'url': f'url/{i}', 'text': SOURCE} for i in range(50)]
        async def open(self, *args):
            return [{'docid': 'MISSING', 'url': '', 'text': 'Document not found for given docid.'}]
    env = LocalSearch.__new__(LocalSearch)
    env.client = Search()
    env.stats = Counter()
    env.search_topk_cap = 1
    env.search_snippet_words = 512
    env.search_snippet_chars = 12000
    env.open_page_words = 4096
    env.open_page_chars = 48000
    env.visited_pages = set()
    env.evidence_documents = []
    env.record_evidence = True
    env.search_skill_context = None
    response = asyncio.run(env.run_action('<function=search><parameter=query>critic</parameter></function>'))
    assert [d['docid'] for d in env.evidence_documents] == ['0']
    assert 'docid: 1\n' not in response['observation']
    asyncio.run(env.run_action('<function=open_page><parameter=docid>MISSING</parameter></function>'))
    assert [d['docid'] for d in env.evidence_documents] == ['0']


def test_ray_batch_retains_decision_mask_including_zero_reward_decision():
    # Execute the production batching method without importing GPU/server modules.
    import ast
    from pathlib import Path
    import torch
    from tensordict import TensorDict
    from verl import DataProto
    import os
    source = Path(__file__).parents[1] / 'verl/experimental/agent_loop/agent_loop.py'
    tree = ast.parse(source.read_text(encoding='utf8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AgentLoopWorkerBase')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_postprocess')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    ns = {'torch': torch, 'np': np, 'TensorDict': TensorDict, 'DataProto': DataProto, 'os': os}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), ns)
    items = []
    for decision in [torch.tensor([[1., 0.]]), None]:
        items.append(SimpleNamespace(prompt_ids=torch.ones(1, 2, dtype=torch.long),
            response_ids=torch.ones(1, 2, dtype=torch.long), response_mask=torch.ones(1, 2),
            input_ids=torch.ones(1, 4, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long),
            position_ids=torch.arange(4).unsqueeze(0), graph_decision_mask=decision,
            graph_edit_credit_mask=torch.zeros(1, 2), reward_score=0., num_turns=1,
            response_logprobs=None, routed_experts=None, process_reward_mask=None, extra_fields={},
            multi_modal_inputs=None, metrics=SimpleNamespace(model_dump=lambda: {})))
    batch = ns['_postprocess'](SimpleNamespace(), items)
    assert batch.batch['graph_decision_mask'].tolist() == [[1., 0.], [0., 0.]]
    assert batch.batch['graph_edit_credit_mask'].sum() == 0
