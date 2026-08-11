"""Shared tool-observation tracking for ContextGraph agent loops."""

from typing import Optional

from .context_graph import ContextGraph, EdgeRelation, NodeType


def record_tool_observation(
    graph: ContextGraph,
    fn_call: dict | None,
    observation: str | None,
) -> Optional[str]:
    """Record a supported tool result as an observation node.

    Keeping this mapping in one place ensures that the main graph and isolated
    branch graphs use identical tool adapters. Unsupported calls (for example
    ``finish`` and graph-management operations) intentionally do not add a
    node.
    """
    if fn_call is None or observation is None:
        return None

    function = fn_call.get("function")
    arguments = fn_call.get("arguments") or {}
    metadata = {"tool": function}
    relation = EdgeRelation.TEMPORAL
    preview_chars = 500

    if function == "search":
        metadata["query"] = arguments.get("query", "")
    elif function == "open_page":
        relation = EdgeRelation.CAUSAL
        preview_chars = 800
    elif function == "action":
        metadata["command"] = arguments.get("command", "")[:100]
        preview_chars = 300
    elif function == "python_exec":
        # ScienceAgentBench exposes python_exec rather than search/open_page.
        # Store a bounded code preview for provenance while the node content
        # carries the execution result returned by the environment.
        metadata["code"] = arguments.get("code", "")[:200]
        preview_chars = 800
    else:
        return None

    return graph.add_node(
        str(observation)[:preview_chars],
        NodeType.OBSERVATION,
        parent_id=graph.active_node_id,
        edge_relation=relation,
        metadata=metadata,
    )
