from agents.context_graph import ContextGraph, NodeType, EdgeRelation
from agents.graph_memory_aids import repeat_note, set_vocabulary
from agents.tool_spec import graph_tool


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
