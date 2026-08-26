from pathlib import Path

from agents.context_graph import ContextGraph, EdgeRelation, NodeType
from agents.graph_observation import record_tool_observation
from agents.prompts_code import create_chat_code


ROOT = Path(__file__).resolve().parents[1]


def test_python_exec_result_adds_observation_and_resets_saturation_clock():
    graph = ContextGraph()
    root_id = graph.add_node("science task", NodeType.QUERY)
    graph.turns_since_last_node_add = 3

    node_id = record_tool_observation(
        graph,
        {
            "function": "python_exec",
            "arguments": {"code": "print('measurement')"},
        },
        "measurement=42\n[output_status] present=yes",
    )

    assert node_id is not None
    node = graph.nodes[node_id]
    assert node.type == NodeType.OBSERVATION
    assert node.content.startswith("measurement=42")
    assert node.metadata == {
        "tool": "python_exec",
        "code": "print('measurement')",
    }
    assert graph.turns_since_last_node_add == 0
    assert any(
        edge.source == root_id
        and edge.target == node_id
        and edge.relation == EdgeRelation.TEMPORAL
        for edge in graph.edges
    )


def test_unsupported_tool_does_not_mutate_graph():
    graph = ContextGraph()
    graph.add_node("science task", NodeType.QUERY)
    initial_nodes = len(graph.nodes)
    initial_edges = len(graph.edges)

    assert record_tool_observation(
        graph,
        {"function": "finish", "arguments": {"message": "done"}},
        "finished",
    ) is None
    assert len(graph.nodes) == initial_nodes
    assert len(graph.edges) == initial_edges


def test_sab_main_and_branch_paths_share_observation_adapter():
    source = (ROOT / "agents/graph_agent_code_isolated.py").read_text(encoding="utf-8")

    assert "from .graph_observation import record_tool_observation" in source
    assert source.count("record_tool_observation(child_graph, fn_call, observation)") == 1
    assert source.count("record_tool_observation(graph, fn_call, observation)") == 1


def test_sab_branch_workflows_require_initial_delegation_only_for_branch_methods():
    react_chat = create_chat_code("analyze data", "code")
    fold_chat = create_chat_code("analyze data", "code_branch")
    graph_chat = create_chat_code("analyze data", "code_graph")
    react_system = react_chat[0]["content"]
    fold_system = fold_chat[0]["content"]
    graph_system = graph_chat[0]["content"]

    branch_contract = "Before MAIN calls `python_exec`, delegate one focused initial branch."
    assert branch_contract not in react_system
    assert branch_contract in fold_system
    assert branch_contract in graph_system
    assert "would generate >5 turns" not in fold_system
    first_call_contract = "Your first tool call must be `branch`, not `python_exec`."
    assert first_call_contract not in react_chat[1]["content"]
    assert first_call_contract in fold_chat[1]["content"]
    assert first_call_contract in graph_chat[1]["content"]


def test_sab_controller_prompt_hides_legacy_graph_tools():
    legacy_system = create_chat_code("analyze data", "code_graph")[0]["content"]
    chat = create_chat_code(
        "analyze data",
        "code_graph",
        expose_graph_tools=False,
    )
    system_prompt = chat[0]["content"]

    assert "`python_exec`" in system_prompt
    assert "`branch`" in system_prompt
    assert "`finish`" in system_prompt
    assert ": merge ----" not in system_prompt
    assert ": add_edge ----" not in system_prompt
    assert ": select ----" not in system_prompt
    assert ": prune ----" not in system_prompt
    assert "[GRAPH MERGE MODE]" in system_prompt
    assert "supplied response schema" in system_prompt
    for tool_name in ("merge", "add_edge", "select", "prune"):
        assert f": {tool_name} ----" in legacy_system
