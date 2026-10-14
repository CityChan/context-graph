import asyncio
from collections import Counter
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts import generate_bcp_graph_sft as generator, build_auto_graph_sft as curator
from tests.test_session_restart import Tokenizer


def arguments(benchmark="bcp"):
    return SimpleNamespace(benchmark=benchmark, graph_protocol="controller", teacher_profile="evaluation",
                           prompt_length=16384, response_length=49152, branch_len=8192, max_turn=30,
                           max_steps=100, turn_max_tokens=2048, workers=1, candidates=1,
                           teacher_model="test-teacher", transport="vllm", temperature=0.7,
                           preflight_only=False)


@pytest.mark.parametrize("benchmark", ["bcp", "scienceworld"])
def test_auto_preserves_evaluation_maintenance_config(benchmark, monkeypatch):
    monkeypatch.setenv("AGENT_BUDGET_MODE", "working_context")
    args = arguments(benchmark)
    plugin = generator.make_config(args).actor_rollout_ref.rollout.plugin
    assert plugin.structured_graph_controller and plugin.controller_owned_tool_formatting
    assert plugin.consolidation_interval == 5 and plugin.auto_prune_max_active == 12
    assert plugin.graph_action_protocol == "controller"
    assert plugin.enable_retrieval_memory
    assert plugin.token_budget_mode == "working_context"
    args.teacher_profile = "framework_assisted_v1"
    with pytest.raises(ValueError, match="evaluation controller"):
        generator.make_config(args)


@pytest.mark.parametrize('benchmark', ['bcp', 'scienceworld'])
def test_full_64k_budget_is_input_plus_output_window(benchmark, monkeypatch):
    from agents.token_budget import TokenBudget
    monkeypatch.setenv('AGENT_BUDGET_MODE', 'working_context')
    args = arguments(benchmark)
    args.prompt_length, args.response_length = 8192, 57344
    config = generator.make_config(args).actor_rollout_ref.rollout
    budget = TokenBudget.from_config(config)
    assert budget.window == 65536 and budget.use_window
    assert config.plugin.val_response_length == 57344


@pytest.mark.parametrize("benchmark", ["bcp", "scienceworld"])
def test_auto_collection_controller_and_curated_last_turns(tmp_path, monkeypatch, benchmark):
    from agents import graph_agent_isolated
    monkeypatch.setenv("AGENT_BUDGET_MODE", "working_context")

    class Env:
        def __init__(self, *args):
            self.stats = Counter()
            self.instance_info = {"problem_statement": "What temperature?"}
            self.is_finish = self.env_fail = False
        async def init_env(self, item):
            pass
        async def run_action(self, response):
            self.stats["action"] += 1
            if self.stats["action"] == 3:
                self.is_finish = True
                self.stats.update(completed=1, environment_natural_end=1, environment_score=100)
                return {"action": "action" if benchmark == "scienceworld" else "finish"}
            return {"observation": "source A: measured 20 C; pressure unknown. " * 10}
        async def get_reward(self, *args):
            return "", 1.0, {}
        def close(self):
            pass

    monkeypatch.setattr(graph_agent_isolated, "select_env", lambda *a: Env)
    actors = iter(['<function=search><parameter=query>temperature</parameter></function>',
                   '<function=open_page><parameter=docid>A</parameter></function>',
                   '<function=finish><parameter=answer>20 C</parameter></function>'] if benchmark == "bcp" else
                  ['<function=action><parameter=command>look around</parameter></function>'] * 3)
    sent = []
    async def create(**kwargs):
        sent.append(copy.deepcopy(kwargs))
        schema = kwargs["extra_body"].get("structured_outputs", {}).get("json")
        content = json.dumps({"action": "merge", "candidate_indices": [0, 1],
                              "summary": "source A: measured 20 C; pressure unknown", "relation": "semantic"}) if schema else next(actors)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                               usage=None, model="test-teacher", id="mock")
    async def close():
        pass
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=close)
    base = generator.TeacherClient
    monkeypatch.setattr(generator, "TeacherClient", lambda *a: base(*a, client=fake))
    monkeypatch.setattr(curator, "TeacherClient", lambda *a: base(*a, client=fake))
    async def preflight(*args):
        pass
    monkeypatch.setattr(generator, "preflight", preflight)
    args = arguments(benchmark)
    config = generator.make_config(args)
    config.actor_rollout_ref.rollout.plugin.consolidation_interval = 2
    root = tmp_path / "run"
    root.mkdir()
    row = {"ability": "LocalSearch", "extra_info": {"query": "What temperature?", "answer": "secret answer", "split": "train"}}
    if benchmark == "scienceworld":
        from scripts.make_scienceworld_data import to_row
        row = to_row({"task_id": "train-boil-0", "task_name": "boil", "variation_idx": 0, "split": "train"}, "scienceworld_graph")
    tokenizer = Tokenizer()
    asyncio.run(generator.collect(args, [row], [0], config, tokenizer, root))
    episode = curator.read_json(root / "episodes/000000-00/episode.json")
    trajectory = curator.read_json(root / episode["trajectory"])
    requests = curator.read_jsonl(root / episode["requests"])
    reason, audit = curator.audit_auto_episode(episode, trajectory, requests)
    assert reason is None, reason
    assert audit["actions"] == (["search", "open_page", "merge", "finish"] if benchmark == "bcp" else ["action", "action", "merge", "action"])
    assert requests[2]["schema"] and requests[2]["sampling"]["temperature"] == 0
    assert requests[0]["schema"] is None
    assert all("secret answer" not in str(r["messages"]) for r in requests)
    bad = copy.deepcopy(requests)
    bad[2]["response"] = bad[2]["response"].replace("unknown", "known")
    assert curator.audit_auto_episode(episode, trajectory, bad)[0] == "unmatched_graph_action"
    bad = copy.deepcopy(episode)
    bad["task_reward"] = 0
    assert curator.audit_auto_episode(bad, trajectory, requests)[0] == "task_not_solved"
    manifest = {"profile": generator.AUTO_PROFILE, "indices": [0], "settings": vars(args), "config": OmegaConf.to_container(config)}
    (root / "manifest.json").write_text(json.dumps(manifest))
    from scripts.diagnose_graph_sft import diagnose
    assert diagnose(root)["tasks"][0]["sampled_graph_tools"] == {"merge": 1}
    async def review(events, *args):
        assert [e["op"] for e in events] == ["merge"]
        return None
    monkeypatch.setattr(curator, "review_edits", review)
    export = SimpleNamespace(run=str(root), output=str(tmp_path / "sft"), review=True,
                             review_max_input=24000, max_length=65536, validation_fraction=0, seed=42)
    asyncio.run(curator.build(export, tokenizer))
    rows = pd.read_parquet(tmp_path / "sft/controller_train.parquet")
    assert len(rows) == 1 and rows.iloc[0]["train_last_response_only"]
    assert rows.iloc[0]["messages"][-1]["content"] == requests[2]["response"]
    assert len(pd.read_parquet(tmp_path / "sft/actor_train.parquet")) == 3

    # The only accepted task may land in validation, as in the live 3x2 smoke.
    monkeypatch.setattr(curator, 'split_for', lambda *a: 'validation')
    export.output = str(tmp_path / 'validation-only')
    with pytest.raises(SystemExit, match='validation only'):
        asyncio.run(curator.build(export, tokenizer))
    from scripts.audit_auto_graph_sft import audit_export
    report = audit_export(export.output)
    assert report['passed'] and not report['training_ready']
    assert report['outputs'] == {'actor_validation': 3, 'controller_validation': 1}
    export.output = str(tmp_path / 'smoke')
    export.smoke = True
    export.reviewer_url = 'http://new-node:8000/v1'
    def current_reviewer(settings, *args):
        assert settings.teacher_url == export.reviewer_url
        return base(settings, *args, client=fake)
    monkeypatch.setattr(curator, 'TeacherClient', current_reviewer)
    asyncio.run(curator.build(export, tokenizer))
    report = curator.read_json(tmp_path / 'smoke/smoke-audit.json')
    assert report['passed'] and not report['training_ready']
    assert not (tmp_path / 'smoke/controller_train.parquet').exists()
    # Smoke does not turn rejected evidence into accepted training labels.
    async def rejected_review(*args):
        return 'evidence_review_failed'
    monkeypatch.setattr(curator, 'review_edits', rejected_review)
    export.output = str(tmp_path / 'rejected-smoke')
    with pytest.raises(SystemExit, match='smoke failed'):
        asyncio.run(curator.build(export, tokenizer))


def test_live_successful_question_hash_stays_in_validation():
    assert curator.split_for('304417167db62052047233127de185b021f35396d7ba37dda6aa99ad0c0f5144', 42, 0.1) == 'validation'


def test_last_response_dataset_masks_rewritten_history(tmp_path):
    import torch
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    # Exercise the real dataset after parquet loading, with a deterministic
    # text template. Old assistant answers occur under replaced observations.
    class Template(Tokenizer):
        pad_token_id = 0
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            if kwargs.get("return_dict"):
                return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
            return ids
    tokenizer = Template()
    messages = [{"role": "user", "content": "Archived evidence marker"},
                {"role": "assistant", "content": "old action"},
                {"role": "user", "content": "Current evidence"},
                {"role": "assistant", "content": "new action"}]
    path = tmp_path / "turns.parquet"
    pd.DataFrame([{"messages": messages, "train_last_response_only": True}]).to_parquet(path)
    dataset = MultiTurnSFTDataset(str(path), tokenizer, config={"max_length": 1024, "pad_mode": "right"})
    result = dataset[0]
    supervised = result["input_ids"][result["loss_mask"].bool()].tolist()
    assert "old action" not in tokenizer.decode(supervised)
    assert "new action" in tokenizer.decode(supervised)
    dataset.truncation = "left"
    with pytest.raises(ValueError, match="truncation=error"):
        dataset[0]


def test_heuristic_operations_are_trace_only_not_sampled_labels():
    from agents.context_graph import ContextGraph, NodeType
    from agents.graph_trace import GraphTraceRecorder
    from agents.graph_controller import graph_action_schema
    graph = ContextGraph()
    graph.add_node("Task", NodeType.QUERY)
    for text in ("obsolete note", "measurement A", "measurement B"):
        graph.add_node(text, NodeType.OBSERVATION)
    trace = GraphTraceRecorder(graph)
    before = trace.capture(graph)
    graph.prune("n2")
    trace.record(graph, before, turn_id=1, source="heuristic", op="auto_prune", args={"node_ids": ["n2"]})
    episode = {"status": "ok", "task_reward": 1, "is_finish": True, "benchmark": "bcp"}
    trajectory = {"graph_action_protocol": "controller", "env_stats": {"natural_finish": 1, "structured_graph_controller": 1}}
    finish = {"messages": [{"role": "user", "content": "Answer now"}], "response": '<function=finish><parameter=answer>A</parameter></function>', "finish_reason": "stop", "schema": None}
    trajectory["graph_trace"] = trace.finalize(graph)
    assert curator.audit_auto_episode(episode, trajectory, [finish])[0] == "no_model_structural_graph_action"
    response = json.dumps({"action": "prune", "candidate_indices": [0], "summary": "", "relation": "semantic"})
    before = trace.capture(graph)
    graph.prune("n3")
    trace.record(graph, before, turn_id=2, source="model", op="prune", args={"node_id": "n3"}, assistant_content=response)
    trajectory["graph_trace"] = trace.finalize(graph)
    request = {**finish, "response": response, "schema": graph_action_schema([0, 1])}
    reason, audit = curator.audit_auto_episode(episode, trajectory, [request, finish])
    assert reason is None, reason
    assert [e["op"] for e in audit["events"]] == ["prune"]
    assert audit["events_by_source"]["heuristic:auto_prune"] == 1
