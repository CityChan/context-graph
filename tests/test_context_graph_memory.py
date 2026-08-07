from agents.context_graph import ContextGraph, EdgeRelation, NodeType


def _archive_branch(
    graph: ContextGraph,
    description: str,
    summary: str,
    preview: str,
    raw_content: str,
) -> tuple[str, str]:
    subtask_id = graph.add_node(
        description,
        NodeType.SUBTASK,
        parent_id=graph.root_id,
        edge_relation=EdgeRelation.DECOMPOSITION,
    )
    child = graph.spawn_child(subtask_id, prefix=f"b{subtask_id}_")
    child.add_node(description, NodeType.QUERY)
    child.add_node(
        preview,
        NodeType.OBSERVATION,
        parent_id=child.root_id,
        metadata={"tool": "open_page", "raw_content": raw_content},
    )
    stats = graph.collapse_child(subtask_id, preserve_archive=True)
    summary_id = graph.add_node(
        summary,
        NodeType.SUMMARY,
        parent_id=subtask_id,
        edge_relation=EdgeRelation.CAUSAL,
    )
    assert graph.attach_archive(summary_id, stats["archive_id"])
    return summary_id, stats["archive_id"]


def test_child_graph_is_archived_without_expanding_active_state():
    graph = ContextGraph()
    graph.add_node("Find the Apollo 11 landing date.", NodeType.QUERY)
    summary_id, archive_id = _archive_branch(
        graph,
        "Research Apollo 11",
        "Apollo 11 landed on the Moon in July 1969.",
        "Apollo mission preview",
        "NASA records state that the lunar module landed on July 20, 1969.",
    )

    subtask_id = graph.archives[archive_id].parent_node_id
    assert graph.nodes[subtask_id].child_graph is None
    assert graph.nodes[summary_id].metadata["archive_id"] == archive_id
    assert graph.nodes[summary_id].metadata["evidence_ids"]
    assert graph.archives[archive_id].evidence[0].content.endswith("July 20, 1969.")
    assert "NASA records" not in graph.to_state_text()


def test_retrieval_drills_from_relevant_summary_to_raw_evidence():
    graph = ContextGraph()
    graph.add_node("Compare dates from the Apollo mission.", NodeType.QUERY)
    _archive_branch(
        graph,
        "Research Apollo landing",
        "Apollo 11 lunar landing chronology.",
        "Short Apollo preview",
        "Primary NASA chronology: Eagle landed on July 20, 1969 at 20:17 UTC.",
    )
    _archive_branch(
        graph,
        "Research bread",
        "Sourdough fermentation notes.",
        "Short cooking preview",
        "The recipe ferments bread dough overnight in a covered bowl.",
    )

    context = graph.retrieve_context(
        "Verify the exact Apollo lunar landing UTC date and time.",
        summary_budget=80,
        evidence_budget=80,
        max_summaries=1,
        max_evidence=1,
    )

    assert "Apollo 11 lunar landing chronology" in context
    assert "July 20, 1969 at 20:17 UTC" in context
    assert "bread dough" not in context
    assert graph.last_retrieval_stats["n_summaries"] == 1
    assert graph.last_retrieval_stats["n_evidence"] == 1


def test_retrieval_respects_separate_working_context_budgets():
    graph = ContextGraph()
    graph.add_node("Find a launch date.", NodeType.QUERY)
    _archive_branch(
        graph,
        "Research launch",
        " ".join(["launch-summary"] * 100),
        "launch preview",
        " ".join(["launch-evidence"] * 100),
    )

    graph.retrieve_context(
        "launch",
        summary_budget=20,
        evidence_budget=25,
        max_summaries=5,
        max_evidence=4,
    )

    assert graph.last_retrieval_stats["summary_tokens"] <= 20
    assert graph.last_retrieval_stats["evidence_tokens"] <= 25


def test_pruning_removes_evidence_from_active_graph_but_not_retrieval():
    graph = ContextGraph()
    graph.add_node("Find the treaty signing date.", NodeType.QUERY)
    evidence_id = graph.add_node(
        "short treaty preview",
        NodeType.OBSERVATION,
        parent_id=graph.root_id,
        metadata={
            "tool": "open_page",
            "raw_content": "The treaty was signed in Paris on 10 February 1763.",
        },
    )
    assert graph.prune(evidence_id)

    context = graph.retrieve_context(
        "What exact date was the treaty signed in Paris?",
        summary_budget=10,
        evidence_budget=40,
        max_summaries=1,
        max_evidence=1,
    )

    assert "10 February 1763" in context
