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
