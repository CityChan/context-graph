"""Graph tool handlers shared by search and code isolated executors."""

from .context_graph import ContextGraph, EdgeRelation, GraphOpResult


def handle_merge(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    node_ids_str = fn_call['arguments'].get('node_ids', '')
    summary = fn_call['arguments'].get('summary', '')
    node_ids = [nid.strip() for nid in node_ids_str.split(',') if nid.strip()]

    if len(node_ids) < 2:
        return GraphOpResult(f"[Error] merge requires at least 2 node IDs (got {len(node_ids)}).\n\n{graph.to_state_text()}", False)

    invalid = [
        nid for nid in node_ids
        if nid not in graph.nodes or not graph.nodes[nid].is_active()
    ]
    if invalid:
        return GraphOpResult(f"[Error] Inactive or unknown node IDs: {invalid}.\n\n{graph.to_state_text()}", False)
    if len(set(node_ids)) != len(node_ids):
        return GraphOpResult(f"[Error] merge node IDs must be unique: {node_ids}.\n\n{graph.to_state_text()}", False)
    if len(node_ids) > 6:
        return GraphOpResult(f"[Error] merge accepts at most 6 node IDs (got {len(node_ids)}).\n\n{graph.to_state_text()}", False)
    if not summary.strip():
        return GraphOpResult(f"[Error] merge requires a non-empty summary.\n\n{graph.to_state_text()}", False)

    merged_id = graph.merge(node_ids, summary)
    if merged_id is None:
        return GraphOpResult(f"[Error] Could not merge nodes {node_ids}.\n\n{graph.to_state_text()}", False)

    return GraphOpResult(f"Merged {node_ids} into [{merged_id}].\n\n{graph.to_state_text()}", True)


def handle_add_edge(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    source = fn_call['arguments'].get('source', '').strip()
    target = fn_call['arguments'].get('target', '').strip()
    relation_str = fn_call['arguments'].get('relation', 'semantic').strip()

    relation_map = {
        'causal': EdgeRelation.CAUSAL,
        'semantic': EdgeRelation.SEMANTIC,
        'temporal': EdgeRelation.TEMPORAL,
    }
    relation = relation_map.get(relation_str, EdgeRelation.SEMANTIC)

    if source not in graph.nodes or not graph.nodes[source].is_active():
        return GraphOpResult(f"[Error] Inactive or unknown source node: {source}.\n\n{graph.to_state_text()}", False)
    if target not in graph.nodes or not graph.nodes[target].is_active():
        return GraphOpResult(f"[Error] Inactive or unknown target node: {target}.\n\n{graph.to_state_text()}", False)

    if not graph.add_edge(source, target, relation):
        return GraphOpResult(f"[Error] Cannot add self-loop or duplicate edge {source} --{relation_str}--> {target}.\n\n{graph.to_state_text()}", False)
    return GraphOpResult(f"Added edge {source} --{relation_str}--> {target}.\n\n{graph.to_state_text()}", True)


def handle_select(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    node_id = fn_call['arguments'].get('node_id', '').strip()

    if node_id not in graph.nodes:
        return GraphOpResult(f"[Error] Unknown node: {node_id}.\n\n{graph.to_state_text()}", False)

    if not graph.select(node_id):
        return GraphOpResult(f"[Error] Cannot select {node_id} (inactive?).\n\n{graph.to_state_text()}", False)

    node = graph.nodes[node_id]
    content_preview = node.content[:500]
    from .graph_memory_aids import neutral
    label = "Active node set to" if neutral() else "Focus shifted to"
    return GraphOpResult(f"{label} [{node_id}] ({node.type.value}).\n\nContent:\n{content_preview}\n\n{graph.to_state_text()}", True)


def handle_prune(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    node_id = fn_call['arguments'].get('node_id', '').strip()

    if node_id not in graph.nodes:
        return GraphOpResult(f"[Error] Unknown node: {node_id}.\n\n{graph.to_state_text()}", False)

    if not graph.prune(node_id):
        return GraphOpResult(f"[Error] Cannot prune {node_id} (root or inactive node?).\n\n{graph.to_state_text()}", False)

    return GraphOpResult(f"Pruned [{node_id}].\n\n{graph.to_state_text()}", True)


GRAPH_OPS = {'merge', 'add_edge', 'select', 'prune'}
