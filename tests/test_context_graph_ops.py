import pytest

from agents.context_graph import ContextGraph, EdgeRelation, GraphOpResult, NodeStatus, NodeType


def _graph_with_observations(count: int = 2):
    graph = ContextGraph()
    root = graph.add_node("question", NodeType.QUERY)
    observations = [
        graph.add_node(
            f"evidence {index}",
            NodeType.OBSERVATION,
            parent_id=root,
            edge_relation=EdgeRelation.TEMPORAL,
        )
        for index in range(count)
    ]
    return graph, root, observations


def test_merge_rejects_folded_nodes_and_does_not_grow_again():
    graph, root, observations = _graph_with_observations()

    summary = graph.merge(observations, "combined evidence")
    assert summary is not None
    assert all(graph.nodes[node_id].status == NodeStatus.FOLDED for node_id in observations)
    assert {node.id for node in graph.active_nodes} == {root, summary}
    assert len(graph.active_edges) == 1

    node_count = len(graph.nodes)
    edge_count = len(graph.edges)
    assert graph.merge(observations, "duplicate merge") is None
    assert len(graph.nodes) == node_count
    assert len(graph.edges) == edge_count


def test_merge_rejects_duplicates_root_and_oversized_requests():
    graph, root, observations = _graph_with_observations(7)

    assert graph.merge([observations[0], observations[0]], "duplicate") is None
    assert graph.merge([root, observations[0]], "includes root") is None
    assert graph.merge(observations, "too many") is None
    assert len(graph.nodes) == 8


def test_merge_rejects_empty_summary_without_mutating_graph():
    graph, _, observations = _graph_with_observations()
    node_count = len(graph.nodes)
    edge_count = len(graph.edges)

    assert graph.merge(observations, "") is None
    assert graph.merge(observations, "   ") is None
    assert len(graph.nodes) == node_count
    assert len(graph.edges) == edge_count
    assert all(graph.nodes[node_id].status == NodeStatus.ACTIVE for node_id in observations)


def test_graph_mutations_require_active_nodes_and_unique_edges():
    graph, _, observations = _graph_with_observations(3)
    assert graph.add_edge(observations[0], observations[1], EdgeRelation.SEMANTIC)
    assert not graph.add_edge(observations[0], observations[1], EdgeRelation.SEMANTIC)
    assert not graph.add_edge(observations[0], observations[0], EdgeRelation.SEMANTIC)

    summary = graph.merge(observations[:2], "summary")
    assert summary is not None
    assert not graph.add_edge(observations[0], observations[2], EdgeRelation.CAUSAL)
    assert not graph.select(observations[0])
    assert not graph.prune(observations[0])


def test_select_rejects_current_focus_without_mutating_counters():
    graph, root, observations = _graph_with_observations()
    initial_operations = graph.operation_count

    assert not graph.select(root)
    assert graph.active_node_id == root
    assert graph.operation_count == initial_operations

    assert graph.select(observations[0])
    assert graph.active_node_id == observations[0]
    assert graph.operation_count == initial_operations + 1


def test_state_text_only_advertises_active_tool_ids():
    graph, _, observations = _graph_with_observations()
    summary = graph.merge(observations, "summary")

    state = graph.to_state_text()
    assert summary in state
    assert f"[{observations[0]}]" not in state
    assert f"[{observations[1]}]" not in state
    assert "Eligible graph-tool node IDs" in state


def test_failed_task_cannot_earn_positive_graph_reward():
    graph, _, observations = _graph_with_observations()
    graph.add_edge(observations[0], observations[1], EdgeRelation.SEMANTIC)
    graph.explicit_op_count = 20

    reward = graph.compute_graph_reward(0.0, lambda_compact=0.2, lambda_cost=0.002)
    assert reward["graph_shaping"] == 0.0
    assert reward["usage_bonus"] == 0.0
    assert reward["graph_reward"] == 0.0

    graph.invalid_op_count = 2
    graph.graph_op_attempt_count = 22
    reward = graph.compute_graph_reward(0.0, lambda_compact=0.2, lambda_cost=0.002)
    assert reward["graph_reward"] == pytest.approx(-0.02)
    assert reward["invalid_op_rate"] == pytest.approx(2 / 22)


def test_success_shaping_is_capped_and_only_successful_ops_are_counted():
    graph, _, observations = _graph_with_observations()
    result = GraphOpResult("ok", True)
    invalid = GraphOpResult("error", False)
    graph.record_graph_op(result.success)
    graph.record_graph_op(invalid.success)
    graph.merge(observations, "summary")

    reward = graph.compute_graph_reward(1.0, lambda_compact=1.0, lambda_cost=0.0)
    assert graph.explicit_op_count == 1
    assert graph.invalid_op_count == 1
    assert graph.graph_op_attempt_count == 2
    assert reward["graph_shaping"] <= 0.1
    assert reward["graph_reward"] == pytest.approx(1.09)


def test_graph_operation_budget_bounds_long_trajectories():
    graph = ContextGraph()
    graph.explicit_op_count = 10
    assert "valid graph operation budget exhausted" in graph.graph_op_budget_error()
    state = graph.to_state_text()
    assert "Graph tools disabled" in state
    assert "Continue with environment actions" in state
    assert "Eligible graph-tool node IDs" not in state

    graph.explicit_op_count = 0
    graph.graph_op_attempt_count = 20
    assert "attempt budget exhausted" in graph.graph_op_budget_error()
