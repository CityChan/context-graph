"""Regression coverage for stale ScienceWorld graph state; no model-quality claims."""
import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from agents.context_graph import ContextGraph, NodeType
from scripts.audit_scienceworld_pair import protocol_check
from scripts.eval_agent_benchmarks import config_for


def test_recent_failure_is_protected_even_with_lower_value():
    graph = ContextGraph(memory_policy="repaired")
    root = graph.add_node("Find the workshop", NodeType.QUERY)
    ids = []
    for step in range(30):
        node = graph.add_node(f"feedback {step}", NodeType.OBSERVATION, parent_id=root)
        ids.append(node)
        graph.update_value(node, -step)
        removed = graph.auto_prune_low_value(max_active=12, keep_recent=8)
        assert not set(removed) & set(ids[-8:])
        assert all(graph.nodes[n].is_active() for n in ids[-8:])
        assert len(graph.active_nodes) <= 12
    assert graph.nodes[ids[0]].is_active()  # old valuable evidence can also survive


def test_protected_observations_make_pruning_cap_soft():
    graph = ContextGraph(memory_policy="repaired")
    root = graph.add_node("task", NodeType.QUERY)
    for i in range(4):
        graph.add_node(str(i), NodeType.OBSERVATION, parent_id=root)
    assert graph.auto_prune_low_value(max_active=2, keep_recent=4) == []


@pytest.mark.parametrize("profile,counts", [("legacy", True), ("turns", False), ("repaired", False)])
def test_profiles_preserve_budgets_and_record_method_differences(profile, counts):
    configs = [config_for("scienceworld", method, 65536, memory_profile=profile)
               for method in ["contextgraph", "foldagent"]]
    for config in configs:
        rollout = config.actor_rollout_ref.rollout
        assert rollout.response_length == 57344
        assert rollout.plugin.scienceworld_max_steps == 100
        assert rollout.plugin.max_turn == 100
        assert rollout.plugin.session_timeout == 3600
        assert rollout.plugin.graph_controller_counts_as_turn == counts
    cg, fa = [x.actor_rollout_ref.rollout.plugin for x in configs]
    assert cg.get("contextgraph_memory_mode", "legacy") == ("repaired" if profile == "repaired" else "legacy")
    assert "auto_prune_keep_recent" not in fa
    manifests = [{"task_ids": ["task"], "config": OmegaConf.to_container(x)} for x in configs]
    assert protocol_check([manifests[0]], [manifests[1]])["matched"]
    manifests[1]["config"]["actor_rollout_ref"]["rollout"]["plugin"]["scienceworld_memory_profile"] = "different"
    assert not protocol_check([manifests[0]], [manifests[1]])["matched"]


def test_prompt_profile_is_a_shared_comparison_constraint():
    configs = [config_for("scienceworld", method, 65536, prompt_profile="focus_v2")
               for method in ["contextgraph", "foldagent"]]
    manifests = [{"task_ids": ["task"], "config": OmegaConf.to_container(x)} for x in configs]
    assert protocol_check([manifests[0]], [manifests[1]])["matched"]
    manifests[1]["config"]["actor_rollout_ref"]["rollout"]["plugin"]["scienceworld_prompt_profile"] = "legacy"
    assert not protocol_check([manifests[0]], [manifests[1]])["matched"]


@pytest.mark.parametrize("method", ["contextgraph", "foldagent"])
def test_default_changes_only_controller_accounting_from_legacy(method):
    default = config_for("scienceworld", method, 65536)
    legacy = config_for("scienceworld", method, 65536, memory_profile="legacy")
    plugin = default.actor_rollout_ref.rollout.plugin
    assert plugin.scienceworld_memory_profile == "turns"
    assert plugin.scienceworld_prompt_profile == "legacy"
    assert plugin.graph_controller_counts_as_turn is False
    # All model, memory, prompt and budget settings must remain identical.
    plugin.graph_controller_counts_as_turn = True
    plugin.scienceworld_memory_profile = "legacy"
    assert OmegaConf.to_container(default) == OmegaConf.to_container(legacy)


def test_discoveryworld_keeps_its_existing_default():
    default = config_for("discoveryworld", "contextgraph", 65536,
                         prompt_profile="discoveryworld_v1")
    explicit = config_for("discoveryworld", "contextgraph", 65536,
                          memory_profile="repaired", prompt_profile="discoveryworld_v1")
    assert OmegaConf.to_container(default) == OmegaConf.to_container(explicit)


def test_repaired_loop_keeps_latest_feedback_and_recent_history(monkeypatch):
    from agents.graph_agent_isolated import process_item
    from tests.test_session_restart import Tokenizer
    from verl import DataProto

    class Simulator:
        def __init__(self, **kwargs): self.steps = 0
        def load(self, **kwargs): pass
        def reset(self): return "This room is called the kitchen.", {"score": 0}
        def get_task_description(self): return "Reach the workshop."
        def step(self, command):
            self.steps += 1
            # The old numeric menu must remain historical once a later action resolves it.
            text = {1: "Ambiguous request: enter a number. 0: open door",
                    2: "The door is now open.",
                    3: "You move to the bathroom.",
                    4: "No known action matches that input."}.get(self.steps, f"Feedback step {self.steps}")
            done = self.steps == 18
            return text, 0, done, {"score": 100 if done else 0}
        def close(self): pass

    class Client:
        def __init__(self): self.calls = []
        async def create_completion(self, ids, **kwargs):
            self.calls.append((''.join(map(chr, ids)), kwargs))
            schema = (kwargs.get("structured_outputs") or {}).get("json", {})
            if "candidate_indices" in schema.get("properties", {}):
                response = json.dumps({"action": "pass", "candidate_indices": [], "summary": "", "relation": "causal"})
            else:
                response = "<function=action><parameter=command>look around</parameter></function>"
            tokens = list(map(ord, response)) + [0]
            return {"choices": [{"message": {"content": response, "raw_output_ids": tokens,
                                               "response_log_probs": [-0.1] * len(tokens)}}]}

    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=Simulator))
    config = config_for("scienceworld", "contextgraph", 65536, memory_profile="repaired")
    config.actor_rollout_ref.rollout.response_length = 1_000_000
    config.actor_rollout_ref.rollout.plugin.val_response_length = 1_000_000
    task = DataProto()
    task.non_tensor_batch = {"ability": np.array(["ScienceWorld@real"], dtype=object),
                             "extra_info": np.array([{"task_name": "synthetic"}], dtype=object),
                             "uid": np.array(["synthetic"], dtype=object)}
    task.meta_info = {"generation_kwargs": {}}
    client = Client()
    output = asyncio.run(process_item(task, SimpleNamespace(config=config, tokenizer=Tokenizer(),
                        llm_client=client, is_train=False, global_step=0)))
    row = output[0].extra_fields
    assert row["env_stats"]["environment_steps"] == 18
    assert row["env_stats"]["turn_budget_used"] == 18
    assert row["env_stats"]["graph_controller_turns"] == 3
    assert row["contextgraph_memory_mode"] == "repaired"
    ordinary = [text for text, kwargs in client.calls if not kwargs.get("structured_outputs")]
    assert "[Latest tool feedback]\nNo known action matches that input." in ordinary[4]
    assert "The door is now open." in ordinary[4]
    assert "You move to the bathroom." in ordinary[4]
    assert "[Historical ContextGraph evidence]" in ordinary[4]
    assert "original pending interaction" in ordinary[4]
    assert "[Archived working-memory payload" not in ordinary[-1]
    # Controller acknowledgements still include graph state; ordinary feedback does not.
    assert "[Latest ContextGraph state]" not in ordinary[-1].rsplit("<user>", 1)[-1]
    latest = None
    prunes = []
    for event in row["graph_trace"]["events"]:
        if event["op"] == "add_observation": latest = event["args"]["node_id"]
        if event["op"] == "auto_prune":
            prunes.append(event)
            assert latest not in event["args"]["node_ids"]
    assert prunes
