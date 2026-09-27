"""Behavioral checks for helpers shared by search and code executors."""
import pytest

from agents.agent_text import clean_response, extract_fn_call, extract_summary
from agents.context_graph import ContextGraph, NodeType, NodeStatus
from agents.graph_operations import handle_merge, handle_add_edge, handle_select, handle_prune


def test_last_tool_call_keeps_multiline_arguments_and_summary():
    text = '<function=search><parameter=query>old</parameter></function><function=return><parameter=message>line one\nline two</parameter></function>'
    assert extract_fn_call(text) == {'function': 'return', 'arguments': {'message': 'line one\nline two'}}
    assert extract_fn_call(None) is None
    assert extract_fn_call('plain text') is None
    assert extract_summary('<summary>old</summary><summary> new\n fact </summary>') == 'new\n fact'
    assert extract_summary('no summary') is None
    assert 'token budget exhausted' in clean_response(None)
    assert clean_response('prefix<function=return>answer') == 'answer'


def graph_fixture():
    graph = ContextGraph()
    root = graph.add_node('question', NodeType.QUERY)
    a = graph.add_node('evidence A', NodeType.OBSERVATION, parent_id=root)
    b = graph.add_node('evidence B', NodeType.OBSERVATION, parent_id=root)
    return graph, root, a, b


@pytest.mark.parametrize('ids,summary', [('duplicate', 'combined'), ('unknown', 'combined'), ('valid', '')])
def test_invalid_merge_leaves_graph_unchanged(ids, summary):
    graph, root, a, b = graph_fixture()
    before = graph.to_state_text()
    node_ids = {'duplicate': f'{a},{a}', 'unknown': f'{a},missing', 'valid': f'{a},{b}'}[ids]
    result = handle_merge(graph, {'arguments': {'node_ids': node_ids, 'summary': summary}})
    assert '[Error]' in str(result)
    assert graph.to_state_text() == before


def test_graph_tools_preserve_state_transitions():
    graph, root, a, b = graph_fixture()
    handle_add_edge(graph, {'arguments': {'source': a, 'target': b}})
    assert '[Error]' in str(handle_add_edge(graph, {'arguments': {'source': a, 'target': b}}))
    handle_select(graph, {'arguments': {'node_id': a}})
    assert graph.active_node_id == a
    handle_merge(graph, {'arguments': {'node_ids': f'{a},{b}', 'summary': 'combined'}})
    assert graph.nodes[a].status == NodeStatus.FOLDED
    assert '[Error]' in str(handle_select(graph, {'arguments': {'node_id': a}}))
    assert '[Error]' in str(handle_prune(graph, {'arguments': {'node_id': root}}))
