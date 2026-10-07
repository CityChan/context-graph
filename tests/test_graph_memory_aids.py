import asyncio
import sys
from types import SimpleNamespace

import numpy as np

from agents.context_graph import ContextGraph, NodeType, EdgeRelation
from agents.graph_memory_aids import commit_evidence, repeat_note, set_vocabulary
from agents.tool_spec import graph_tool
from envs.scienceworld_env import ScienceWorldEnv


def item(extra):
    return SimpleNamespace(non_tensor_batch={"extra_info": np.array([extra], dtype=object)})


def _select_text():
    select = next(t for t in graph_tool() if t["function"]["name"] == "select")
    return select["function"]["description"] + select["function"]["parameters"]["properties"]["node_id"]["description"]


def test_neutral_vocabulary_changes_only_wording():
    graph = ContextGraph()
    graph.add_node("look around", NodeType.OBSERVATION)
    try:
        set_vocabulary("active")
        assert "focus" not in _select_text().lower()
        assert "active=[" in graph.to_state_text() and "focus=[" not in graph.to_state_text()
        set_vocabulary("focus")
        assert "focus" in _select_text().lower() and "focus=[" in graph.to_state_text()
    finally:
        set_vocabulary("focus")


def test_repeat_note_counts_identical_action_results_only():
    graph = ContextGraph()
    meta = lambda cmd, obs: {"tool": "action", "command": cmd, "raw_content": obs}
    a = graph.add_node("already open", NodeType.OBSERVATION, metadata=meta("open closet", "The closet is already open."))
    assert repeat_note(graph, a) == ""
    b = graph.add_node("already open", NodeType.OBSERVATION, metadata=meta("Open  Closet", "The closet is already open."))
    note = repeat_note(graph, b)
    assert "1 time(s)" in note and a in note
    c = graph.add_node("opened", NodeType.OBSERVATION, metadata=meta("open closet", "The closet is now open."))
    assert repeat_note(graph, c) == ""
    d = graph.add_node("thinking", NodeType.OBSERVATION, metadata={"tool": "think", "raw_content": "x"})
    assert repeat_note(graph, d) == ""


def test_commit_evidence_quotes_graph_lines_about_the_target():
    graph = ContextGraph()
    meta = lambda cmd, obs: {"tool": "action", "command": cmd, "raw_content": obs}
    n1 = graph.add_node("look", NodeType.OBSERVATION, metadata=meta(
        "look around", "This room is called the greenhouse. In it, you see:\n\ta flower pot 4 (containing a peach tree)\n\ta bee hive"))
    graph.add_node("check", NodeType.OBSERVATION, metadata=meta("focus on flower pot 4", "[Commit check] flower pot 4"))
    text = commit_evidence(graph, "Flower Pot 4")
    assert n1 in text and "peach tree" in text and "[Commit check]" not in text and "bee hive" not in text
    assert "peach tree" in commit_evidence(graph, "the flower pot 4")  # leading articles are ignored
    assert "peach tree" in commit_evidence(graph, "old tree")  # falls back to the head noun
    assert "(`look around`)" in text
    assert "No stored observation" in commit_evidence(graph, "unicorn")


def test_commit_check_requires_repeating_focus_before_stepping(monkeypatch, tmp_path):
    stepped = []

    class Simulator:
        def __init__(self, **kwargs):
            pass
        def load(self, **kwargs):
            pass
        def reset(self):
            return "room", {"score": 0}
        def get_task_description(self):
            return "Your task is to find a plant. First, focus on the thing."
        def step(self, command):
            stepped.append(command)
            return f"did {command}", 0, False, {"score": 0, "moves": len(stepped)}
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=Simulator))
    act = lambda c: asyncio.run(env.run_action(f"<function=action><parameter=command>{c}</parameter></function>"))
    for enabled in (False, True):
        stepped.clear()
        env = ScienceWorldEnv(SimpleNamespace(plugin=SimpleNamespace(scienceworld_commit_check=enabled)), None, "x")
        env.commit_evidence = (lambda target: f"evidence for {target}") if enabled else None
        asyncio.run(env.init_env(item({"task_name": "find-plant", "tool_log": str(tmp_path / f"{enabled}.jsonl")})))
        first = act("focus on flower pot 4")["observation"]
        if not enabled:
            assert stepped == ["focus on flower pot 4"]
            continue
        assert stepped == [] and "[Commit check]" in first and "find a plant" in first and "evidence for flower pot 4" in first
        act("focus on peach tree")
        act("look around")
        act("Focus  on peach tree")
        act("focus on flower pot 4")
        assert stepped == ["look around", "Focus  on peach tree", "focus on flower pot 4"]
        assert env.stats["commit_checks"] == 2 and env.stats["commit_confirmed"] == 2
        assert env.stats["environment_steps"] == 3
