from agents.context_graph import ContextGraph, NodeType
from agents.graph_trace import GraphTraceRecorder
from scripts.validate_contextgraph_traces import validate_results


def _valid_result():
    graph = ContextGraph()
    root = graph.add_node("question", NodeType.QUERY)
    node = graph.add_node("evidence", NodeType.OBSERVATION, parent_id=root)
    recorder = GraphTraceRecorder(graph)
    before = recorder.capture(graph)
    assert graph.prune(node)
    graph.record_graph_op(True)
    recorder.record(
        graph, before, turn_id=1, source="model", op="prune",
        args={"node_id": node}, success=True,
    )
    return {
        "task_id": "task-1",
        "status": "success",
        "env_stats": {"graph_explicit_ops": 1},
        "graph_trace": recorder.finalize(graph),
    }


def test_validate_results_accepts_trace_and_counter_alignment():
    counters, errors = validate_results([_valid_result()])
    assert not errors
    assert counters["valid_traces"] == 1
    assert counters["explicit_valid_model_ops"] == 1


def test_validate_results_rejects_missing_trace_and_counter_mismatch():
    result = _valid_result()
    result["env_stats"]["graph_explicit_ops"] = 2
    _, errors = validate_results([result])
    assert any("does not match" in error for error in errors)

    _, errors = validate_results([{"task_id": "missing", "status": "success"}])
    assert errors
