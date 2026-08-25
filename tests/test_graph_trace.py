from agents.context_graph import ContextGraph, EdgeRelation, NodeType
from agents.graph_trace import GraphTraceRecorder, snapshot_hash, validate_graph_trace


def _record_observation(graph, recorder, parent, content, turn):
    before = recorder.capture(graph)
    node_id = graph.add_node(
        content,
        NodeType.OBSERVATION,
        parent_id=parent,
        edge_relation=EdgeRelation.TEMPORAL,
        metadata={"raw_content": content * 2},
    )
    recorder.record(
        graph, before, turn_id=turn, source="environment",
        op="add_observation", args={"node_id": node_id}, success=True,
    )
    return node_id


def test_graph_trace_replays_multiple_sources_and_compacts_raw_metadata():
    graph = ContextGraph()
    root = graph.add_node("question", NodeType.QUERY)
    recorder = GraphTraceRecorder(graph)
    first = _record_observation(graph, recorder, root, "first evidence", 1)
    second = _record_observation(graph, recorder, root, "second evidence", 2)

    before = recorder.capture(graph)
    assert graph.add_edge(first, second, EdgeRelation.SEMANTIC)
    graph.record_graph_op(True)
    recorder.record(
        graph, before, turn_id=3, source="model", op="add_edge",
        args={"source": first, "target": second, "relation": "semantic"},
        success=True,
        decision_context={
            "mode": "test_context",
            "graph_hash": "frozen",
            "candidates": [{"index": 0, "node_id": first}],
        },
    )

    trace = recorder.finalize(graph)
    valid, errors, metrics = validate_graph_trace(trace)
    assert valid, errors
    assert metrics["explicit_valid_model_ops"] == 1
    assert metrics["structural_model_ops"] == 1
    assert metrics["quality_score"] == 1.0
    assert trace["events"][2]["decision_context"]["mode"] == "test_context"
    node = trace["events"][0]["after_state"]["nodes"][1]
    assert "raw_content" not in node["metadata"]
    assert node["metadata"]["raw_content_chars"] == len("first evidence" * 2)


def test_graph_trace_flags_redundant_select_and_render_tampering():
    graph = ContextGraph()
    root = graph.add_node("question", NodeType.QUERY)
    recorder = GraphTraceRecorder(graph)
    before = recorder.capture(graph)
    assert graph.select(root)
    graph.record_graph_op(True)
    recorder.record(
        graph, before, turn_id=1, source="model", op="select",
        args={"node_id": root}, success=True,
    )
    trace = recorder.finalize(graph)

    valid, errors, metrics = validate_graph_trace(trace)
    assert valid, errors
    assert metrics["redundant_model_ops"] == 1
    assert metrics["quality_score"] == 0.0

    trace["events"][0]["rendered_before"] += " tampered"
    valid, errors, _ = validate_graph_trace(trace)
    assert not valid
    assert any("rendered before hash mismatch" in error for error in errors)


def test_graph_trace_audits_controller_index_to_node_mapping():
    graph = ContextGraph()
    root = graph.add_node("question", NodeType.QUERY)
    recorder = GraphTraceRecorder(graph)
    first = _record_observation(graph, recorder, root, "first evidence", 1)
    second = _record_observation(graph, recorder, root, "second evidence", 2)

    before = recorder.capture(graph)
    summary = "The two observations jointly support the answer."
    merged = graph.merge([first, second], summary)
    assert merged is not None
    graph.record_graph_op(True)
    recorder.record(
        graph,
        before,
        turn_id=3,
        source="model",
        op="merge",
        args={"node_ids": f"{first},{second}", "summary": summary},
        success=True,
        decision_context={
            "mode": "controller_merge",
            "graph_hash": snapshot_hash(before),
            "candidates": [
                {"index": 0, "node_id": first},
                {"index": 1, "node_id": second},
            ],
            "decision": {
                "candidate_indices": [0, 1],
                "summary": summary,
            },
        },
    )

    trace = recorder.finalize(graph)
    valid, errors, metrics = validate_graph_trace(trace)
    assert valid, errors
    assert metrics["productive_model_ops"] == 1

    trace["events"][-1]["decision_context"]["decision"]["candidate_indices"] = [1, 0]
    valid, errors, _ = validate_graph_trace(trace)
    assert not valid
    assert any("candidate mapping" in error for error in errors)
