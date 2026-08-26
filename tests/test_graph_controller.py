import json

import pytest

from agents.context_graph import ContextGraph, EdgeRelation, NodeType
from agents.graph_controller import GraphActionController, GraphControllerError
from agents.prompts import create_chat


def _graph_with_evidence():
    graph = ContextGraph(namespace_prefix="n")
    root = graph.add_node("task", NodeType.QUERY)
    graph.add_node(
        "apple is on the counter",
        NodeType.OBSERVATION,
        parent_id=root,
        edge_relation=EdgeRelation.TEMPORAL,
    )
    graph.add_node(
        "the apple must be cooled",
        NodeType.OBSERVATION,
        parent_id=root,
        edge_relation=EdgeRelation.TEMPORAL,
    )
    return graph


def test_controller_exposes_indices_and_resolves_frozen_legal_ids():
    graph = _graph_with_evidence()
    controller = GraphActionController()
    snapshot = controller.snapshot(graph)

    assert [candidate.index for candidate in snapshot.candidates] == [0, 1]
    assert [candidate.node_id for candidate in snapshot.candidates] == ["n2", "n3"]
    schema = controller.merge_schema(snapshot)
    assert schema["properties"]["candidate_indices"]["items"]["enum"] == [0, 1]
    assert "uniqueItems" not in schema["properties"]["candidate_indices"]

    args = controller.resolve_merge(
        graph,
        snapshot,
        json.dumps({
            "candidate_indices": [1, 0],
            "summary": "The apple must be found and cooled.",
        }),
    )
    assert args == {
        "node_ids": "n3,n2",
        "summary": "The apple must be found and cooled.",
    }

    duplicate_args = controller.resolve_merge(
        graph,
        snapshot,
        json.dumps({
            "candidate_indices": [1, 1, 0],
            "summary": "The same evidence selection is canonicalized.",
        }),
    )
    assert duplicate_args["node_ids"] == "n3,n2"


def test_controller_exposes_and_resolves_complete_action_space():
    graph = _graph_with_evidence()
    controller = GraphActionController()
    snapshot = controller.snapshot(graph)
    schema = controller.action_schema(snapshot, allow_pass=True)

    assert schema["properties"]["action"]["enum"] == [
        "merge", "prune", "add_edge", "select", "pass",
    ]
    assert "oneOf" not in schema
    assert "uniqueItems" not in schema["properties"]["candidate_indices"]

    def resolve(action, indices, summary="", relation="semantic", allow_pass=False):
        return controller.resolve_action(
            graph,
            snapshot,
            json.dumps({
                "action": action,
                "candidate_indices": indices,
                "summary": summary,
                "relation": relation,
            }),
            allow_pass=allow_pass,
        )

    assert resolve("merge", [0, 1], "combined evidence") == {
        "function": "merge",
        "arguments": {
            "node_ids": "n2,n3",
            "summary": "combined evidence",
        },
    }
    assert resolve("prune", [0]) == {
        "function": "prune", "arguments": {"node_id": "n2"},
    }
    assert resolve("select", [1]) == {
        "function": "select", "arguments": {"node_id": "n3"},
    }
    assert resolve("add_edge", [1, 0], relation="causal") == {
        "function": "add_edge",
        "arguments": {
            "source": "n3", "target": "n2", "relation": "causal",
        },
    }
    assert resolve("pass", [], allow_pass=True) == {
        "function": "pass", "arguments": {},
    }


def test_controller_rejects_action_specific_invalid_fields():
    graph = _graph_with_evidence()
    controller = GraphActionController()
    snapshot = controller.snapshot(graph)

    def response(action, indices, summary="", relation="semantic"):
        return json.dumps({
            "action": action,
            "candidate_indices": indices,
            "summary": summary,
            "relation": relation,
        })

    with pytest.raises(GraphControllerError, match="pass is illegal"):
        controller.resolve_action(
            graph, snapshot, response("pass", []), allow_pass=False,
        )
    with pytest.raises(GraphControllerError, match="exactly 2"):
        controller.resolve_action(
            graph, snapshot, response("add_edge", [0]), allow_pass=False,
        )
    with pytest.raises(GraphControllerError, match="distinct"):
        controller.resolve_action(
            graph, snapshot, response("add_edge", [0, 0]), allow_pass=False,
        )
    with pytest.raises(GraphControllerError, match="exactly 1"):
        controller.resolve_action(
            graph, snapshot, response("prune", [0, 1]), allow_pass=False,
        )


def test_controller_action_prompt_discloses_legality_and_all_actions():
    graph = _graph_with_evidence()
    controller = GraphActionController()
    prompt = controller.action_prompt(
        controller.snapshot(graph), turn_id=5, allow_pass=False,
    )

    assert "[GRAPH ACTION MODE turn=5]" in prompt
    assert all(action in prompt for action in (
        "merge", "prune", "add_edge", "select", "pass",
    ))
    assert "pass is currently illegal" in prompt


def test_controller_rejects_stale_snapshot_and_invalid_decision():
    graph = _graph_with_evidence()
    controller = GraphActionController()
    snapshot = controller.snapshot(graph)

    with pytest.raises(GraphControllerError, match="2 to 6"):
        controller.resolve_merge(
            graph,
            snapshot,
            '{"candidate_indices":[0],"summary":"too small"}',
        )

    with pytest.raises(GraphControllerError, match="two unique"):
        controller.resolve_merge(
            graph,
            snapshot,
            '{"candidate_indices":[0,0],"summary":"duplicate only"}',
        )

    graph.add_node("new evidence", NodeType.OBSERVATION)
    with pytest.raises(GraphControllerError, match="graph changed"):
        controller.resolve_merge(
            graph,
            snapshot,
            '{"candidate_indices":[0,1],"summary":"stale"}',
        )


def test_interactive_prompt_hides_graph_xml_tools_in_controller_mode():
    legacy = create_chat("cool the apple", "alfworld_graph")
    controlled = create_chat(
        "cool the apple",
        "alfworld_graph",
        expose_graph_tools=False,
    )
    assert "Merge multiple context nodes" in legacy[0]["content"]
    assert "Merge multiple context nodes" not in controlled[0]["content"]
    assert "Execute an action in the household environment" in controlled[0]["content"]


def test_search_prompt_removes_normal_mode_graph_instructions_for_controller():
    legacy = create_chat("identify the person", "search_graph")
    controlled = create_chat(
        "identify the person",
        "search_graph",
        expose_graph_tools=False,
    )

    legacy_text = "\n".join(turn["content"] for turn in legacy)
    controlled_text = "\n".join(turn["content"] for turn in controlled)

    assert "<function=merge>" in legacy_text
    assert "<function=merge>" not in controlled_text
    assert "<function=prune>" not in controlled_text
    assert "Use merge/prune/add_edge" not in controlled_text
    assert "[GRAPH ACTION MODE]" in controlled_text
    assert "controller-owned" in controlled_text
    assert ": search ----" in controlled_text
    assert ": branch ----" in controlled_text


def test_controller_requires_budget_for_a_complete_structured_decision():
    controller = GraphActionController(min_completion_tokens=256)

    assert controller.has_completion_budget(320, protected_tokens=64)
    assert not controller.has_completion_budget(319, protected_tokens=64)


def test_controller_rejects_too_small_completion_budget_configuration():
    with pytest.raises(ValueError, match="min_completion_tokens"):
        GraphActionController(min_completion_tokens=9)
