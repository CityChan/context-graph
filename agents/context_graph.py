"""ContextGraph: Dynamic graph-based working context management for LLM agents.

Working context is modeled as a dynamic graph rather than a linear sequence.
Nodes represent context units (observations, actions, subtasks, summaries)
and edges represent relations (temporal, causal, decomposition, semantic).

This generalizes FoldAgent's tree-structured branching to a full graph with:
- Cross-branch connections (edges between any nodes)
- Merge operations (combine multiple nodes into summaries)
- Split operations (decompose nodes into subtasks)
- Pruning (remove low-value context)
- Value-based context selection for token budget management
"""

from __future__ import annotations
import heapq
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


_SEQ_RE = re.compile(r'(\d+)$')
_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]*|[\u4e00-\u9fff]")
_STOP_TERMS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "this", "that", "it", "as", "at",
    "from", "by", "about", "what", "which", "who", "when", "where", "how",
    "function", "parameter", "search", "open_page", "action",
}


def _seq_key(nid: str) -> int:
    """Extract trailing integer from a node ID for stable sorting.

    Handles both single-prefix IDs ('n5') and multi-char prefix IDs ('b0_12')
    used by spawned subgraphs.
    """
    m = _SEQ_RE.search(nid)
    return int(m.group(1)) if m else 0


def _lexical_terms(text: str) -> set[str]:
    return {
        term.lower()
        for term in _TERM_RE.findall(text or "")
        if len(term) > 1 and term.lower() not in _STOP_TERMS
    }


class NodeType(str, Enum):
    QUERY = "query"
    OBSERVATION = "observation"
    ACTION = "action"
    SUBTASK = "subtask"
    SUMMARY = "summary"


class NodeStatus(str, Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    PRUNED = "pruned"


class EdgeRelation(str, Enum):
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    DECOMPOSITION = "decomp"
    SEMANTIC = "semantic"
    MERGE_SOURCE = "merge_src"


@dataclass
class ContextNode:
    id: str
    type: NodeType
    content: str
    status: NodeStatus = NodeStatus.ACTIVE
    value: float = 0.0
    token_count: int = 0
    depth: int = 0
    metadata: dict = field(default_factory=dict)
    # Optional nested subgraph for hierarchical isolation. SUBTASK nodes
    # may host their own ContextGraph (the branch agent's local view).
    # Set to None to release the subgraph's memory after collapse.
    child_graph: Optional[ContextGraph] = None

    def is_active(self) -> bool:
        return self.status == NodeStatus.ACTIVE


@dataclass
class ContextEdge:
    source: str
    target: str
    relation: EdgeRelation
    weight: float = 1.0


class GraphOpResult(str):
    """String-compatible graph-tool result with an explicit success flag."""

    def __new__(cls, message: str, success: bool):
        result = super().__new__(cls, message)
        result.success = success
        return result


@dataclass
class ArchivedEvidence:
    """Immutable evidence retained outside the model's working context."""

    id: str
    source_node_id: str
    type: NodeType
    content: str
    metadata: dict = field(default_factory=dict)
    sequence: int = 0


@dataclass
class ArchivedSubgraph:
    """A detached child graph addressable through a parent summary node."""

    id: str
    parent_node_id: str
    evidence: list[ArchivedEvidence] = field(default_factory=list)
    edges: list[ContextEdge] = field(default_factory=list)
    summary_node_id: Optional[str] = None


class ContextGraph:
    """Dynamic graph for managing agent working context.

    Graph-MDP state: s_t = (observation, graph, active_node)
    Actions: add_node, add_edge, merge, split, select, traverse, prune
    Reward: r_task + lambda1 * r_graph - lambda2 * r_cost
    """

    def __init__(
        self,
        tokenizer=None,
        parent: Optional[ContextGraph] = None,
        namespace_prefix: str = "n",
    ):
        self.nodes: dict[str, ContextNode] = {}
        self.edges: list[ContextEdge] = []
        self.active_node_id: Optional[str] = None
        self.root_id: Optional[str] = None
        self.operation_count: int = 0
        # Separate counter for LLM-initiated graph ops only (merge/prune/select/add_edge
        # tool calls). Auto-heuristic ops (auto_prune, auto_connect, auto_merge, add_node
        # for observations) do NOT count here. cost_penalty uses this, not operation_count.
        self.explicit_op_count: int = 0
        self.graph_op_attempt_count: int = 0
        self.invalid_op_count: int = 0
        self.tokenizer = tokenizer
        self._node_counter = 0
        self._archive_counter = 0
        # Archived subgraphs are deliberately kept outside ``nodes``. They do
        # not appear in graph state or working-context traversal, but remain
        # recoverable through summary -> archive pointers during retrieval.
        self.archives: dict[str, ArchivedSubgraph] = {}
        self.last_retrieval_stats: dict[str, Any] = {}
        # Hierarchical fields (used by isolated graph_agent variant)
        self.parent: Optional[ContextGraph] = parent
        self.namespace_prefix: str = namespace_prefix
        # Forced-consolidation tracking: counts main-loop turns since the last
        # time a node was added. Used by env to decide whether `pass` is a
        # valid response at a consolidation checkpoint (saturation criterion).
        self.turns_since_last_node_add: int = 0

    def _next_id(self, prefix: Optional[str] = None) -> str:
        self._node_counter += 1
        return f"{prefix or self.namespace_prefix}{self._node_counter}"

    def record_graph_op(self, success: bool) -> None:
        """Record one LLM-requested graph operation attempt."""
        self.graph_op_attempt_count += 1
        if success:
            self.explicit_op_count += 1
        else:
            self.invalid_op_count += 1

    def graph_op_budget_error(
        self,
        max_valid_ops: int = 10,
        max_attempts: int = 20,
    ) -> Optional[str]:
        """Return a diagnostic when the per-trajectory graph budget is spent."""
        if self.graph_op_attempt_count >= max_attempts:
            return f"graph operation attempt budget exhausted ({max_attempts})"
        if self.explicit_op_count >= max_valid_ops:
            return f"valid graph operation budget exhausted ({max_valid_ops})"
        return None

    def spawn_child(self, parent_node_id: str, prefix: str) -> ContextGraph:
        """Create an isolated child ContextGraph attached to a SUBTASK node.

        The child has its own root, node ID space (prefixed), active focus,
        and operation count. The parent's ``to_state_text`` / ``to_context_text``
        do NOT include child graph contents — that is the whole point of
        spatial isolation.

        Returns the new child graph (caller is expected to ``add_node`` a root
        on it that mirrors the subtask description).
        """
        if parent_node_id not in self.nodes:
            raise KeyError(f"parent_node_id {parent_node_id} not in graph")
        child = ContextGraph(
            tokenizer=self.tokenizer,
            parent=self,
            namespace_prefix=prefix,
        )
        self.nodes[parent_node_id].child_graph = child
        return child

    def collapse_child(
        self,
        parent_node_id: str,
        preserve_archive: bool = False,
    ) -> Optional[dict]:
        """Detach a child graph and return summary statistics.

        Called after a branch returns and its findings have been folded into
        a SUMMARY node on the parent. When ``preserve_archive`` is true, an
        immutable snapshot is retained outside the active graph. This keeps
        the model-facing working context small without making compression
        irreversible.
        """
        node = self.nodes.get(parent_node_id)
        if node is None or node.child_graph is None:
            return None
        child = node.child_graph
        stats = {
            'n_total': len(child.nodes),
            'n_active': len(child.active_nodes),
            'n_observations': sum(1 for n in child.nodes.values() if n.type == NodeType.OBSERVATION),
            'n_summaries': sum(1 for n in child.nodes.values() if n.type == NodeType.SUMMARY),
            'n_edges': len(child.edges),
            'ops': child.operation_count,
        }
        if preserve_archive:
            self._archive_counter += 1
            archive_id = f"a{self._archive_counter}"
            evidence = []
            for sequence, child_node in enumerate(
                sorted(child.nodes.values(), key=lambda n: _seq_key(n.id))
            ):
                if child_node.type not in (
                    NodeType.OBSERVATION,
                    NodeType.ACTION,
                    NodeType.SUMMARY,
                ):
                    continue
                raw_content = child_node.metadata.get(
                    "raw_content", child_node.content
                )
                evidence.append(
                    ArchivedEvidence(
                        id=f"{archive_id}:{child_node.id}",
                        source_node_id=child_node.id,
                        type=child_node.type,
                        content=raw_content,
                        metadata={
                            key: value
                            for key, value in child_node.metadata.items()
                            if key != "raw_content"
                        },
                        sequence=sequence,
                    )
                )
            self.archives[archive_id] = ArchivedSubgraph(
                id=archive_id,
                parent_node_id=parent_node_id,
                evidence=evidence,
                edges=[
                    ContextEdge(e.source, e.target, e.relation, e.weight)
                    for e in child.edges
                ],
            )
            node.metadata["archive_id"] = archive_id
            stats["archive_id"] = archive_id
            stats["n_archived_evidence"] = len(evidence)
        node.child_graph = None
        return stats

    def attach_archive(self, summary_node_id: str, archive_id: str) -> bool:
        """Create a cross-layer pointer from a summary to archived evidence."""
        summary = self.nodes.get(summary_node_id)
        archive = self.archives.get(archive_id)
        if summary is None or archive is None:
            return False
        summary.metadata["archive_id"] = archive_id
        summary.metadata["evidence_ids"] = [item.id for item in archive.evidence]
        archive.summary_node_id = summary_node_id
        return True

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

    def _truncate_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self._count_tokens(text) <= max_tokens:
            return text
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            try:
                return self.tokenizer.decode(token_ids[:max_tokens])
            except Exception:
                pass
        words = text.split()
        if words:
            return " ".join(words[:max_tokens])
        return text[: max_tokens * 4]

    # ── Core Graph Operations ──

    def add_node(
        self,
        content: str,
        node_type: NodeType,
        parent_id: Optional[str] = None,
        edge_relation: EdgeRelation = EdgeRelation.TEMPORAL,
        metadata: dict = None,
    ) -> str:
        """Add a new context node. Optionally connect to parent."""
        node_id = self._next_id()
        depth = 0
        if parent_id and parent_id in self.nodes:
            depth = self.nodes[parent_id].depth + 1

        node = ContextNode(
            id=node_id,
            type=node_type,
            content=content,
            status=NodeStatus.ACTIVE,
            value=0.0,
            token_count=self._count_tokens(content),
            depth=depth,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        # New content arrived -> reset the consolidation saturation clock.
        self.turns_since_last_node_add = 0

        if parent_id and parent_id in self.nodes:
            self.add_edge(parent_id, node_id, edge_relation)

        if self.root_id is None:
            self.root_id = node_id
            self.active_node_id = node_id

        self.operation_count += 1
        return node_id

    def add_edge(
        self, source: str, target: str, relation: EdgeRelation, weight: float = 1.0
    ) -> bool:
        """Add an active-to-active edge; reject self-loops and duplicates."""
        if source not in self.nodes or target not in self.nodes or source == target:
            return False
        if not self.nodes[source].is_active() or not self.nodes[target].is_active():
            return False
        for e in self.edges:
            if e.source == source and e.target == target and e.relation == relation:
                return False
        self.edges.append(ContextEdge(source, target, relation, weight))
        self.operation_count += 1
        return True

    def merge(
        self,
        node_ids: list[str],
        summary: str,
        max_nodes: int = 6,
    ) -> Optional[str]:
        """Merge multiple nodes into a single summary node.

        Source nodes are marked as FOLDED. The new summary node inherits
        incoming edges from the merged nodes.
        """
        unique_ids = list(dict.fromkeys(node_ids))
        valid_ids = [
            nid for nid in unique_ids
            if nid in self.nodes
            and nid != self.root_id
            and self.nodes[nid].is_active()
        ]
        if len(valid_ids) != len(unique_ids) or not 2 <= len(valid_ids) <= max_nodes:
            return None

        inherited_edges = [
            (edge.source, edge.relation, edge.weight)
            for edge in self.edges
            if edge.target in valid_ids
            and edge.source not in valid_ids
            and edge.source in self.nodes
            and self.nodes[edge.source].is_active()
            and edge.relation != EdgeRelation.MERGE_SOURCE
        ]

        merged_id = self.add_node(
            content=summary,
            node_type=NodeType.SUMMARY,
            metadata={"merged_from": valid_ids},
        )
        # Set depth to min of merged nodes
        depths = [self.nodes[nid].depth for nid in valid_ids]
        self.nodes[merged_id].depth = min(depths) if depths else 0

        for nid in valid_ids:
            self.add_edge(nid, merged_id, EdgeRelation.MERGE_SOURCE)

        # Inherit incoming edges from merged nodes
        for source, relation, weight in inherited_edges:
            self.add_edge(source, merged_id, relation, weight)

        for nid in valid_ids:
            self.nodes[nid].status = NodeStatus.FOLDED

        # Folded history remains recoverable through ``merged_from`` metadata,
        # but it must not inflate or connect the model-facing working graph.
        self.edges = [
            edge for edge in self.edges
            if (
                self.nodes[edge.source].is_active()
                and self.nodes[edge.target].is_active()
            )
            or (
                edge.relation == EdgeRelation.MERGE_SOURCE
                and edge.source in valid_ids
                and edge.target == merged_id
            )
        ]

        if self.active_node_id in valid_ids:
            self.active_node_id = merged_id

        return merged_id

    def split(self, node_id: str, sub_descriptions: list[str]) -> list[str]:
        """Split a node into multiple subtask nodes connected by DECOMPOSITION edges."""
        if node_id not in self.nodes:
            return []

        sub_ids = []
        for desc in sub_descriptions:
            sub_id = self.add_node(
                content=desc,
                node_type=NodeType.SUBTASK,
                parent_id=node_id,
                edge_relation=EdgeRelation.DECOMPOSITION,
            )
            sub_ids.append(sub_id)
        return sub_ids

    def select(self, node_id: str) -> bool:
        """Set the active focus to a specific node."""
        if node_id not in self.nodes:
            return False
        if not self.nodes[node_id].is_active():
            return False
        self.active_node_id = node_id
        self.operation_count += 1
        return True

    def traverse(
        self, direction: str = "children", relation_filter: Optional[str] = None
    ) -> list[str]:
        """Get neighbors of the active node along edges."""
        if self.active_node_id is None:
            return []

        results = []
        for edge in self.edges:
            if direction == "children" and edge.source == self.active_node_id:
                if relation_filter is None or edge.relation.value == relation_filter:
                    if self.nodes.get(edge.target, ContextNode("", NodeType.QUERY, "")).status != NodeStatus.PRUNED:
                        results.append(edge.target)
            elif direction == "parents" and edge.target == self.active_node_id:
                if relation_filter is None or edge.relation.value == relation_filter:
                    if self.nodes.get(edge.source, ContextNode("", NodeType.QUERY, "")).status != NodeStatus.PRUNED:
                        results.append(edge.source)
        return results

    def prune(self, node_id: str) -> bool:
        """Mark a node as pruned (excluded from active context)."""
        if (
            node_id not in self.nodes
            or node_id == self.root_id
            or not self.nodes[node_id].is_active()
        ):
            return False
        self.nodes[node_id].status = NodeStatus.PRUNED
        # Move active focus to parent if pruned node was active
        if self.active_node_id == node_id:
            parents = [e.source for e in self.edges if e.target == node_id]
            self.active_node_id = parents[0] if parents else self.root_id
        self.operation_count += 1
        return True

    def update_value(self, node_id: str, value: float):
        """Update the estimated usefulness of a node."""
        if node_id in self.nodes:
            self.nodes[node_id].value = value

    # ── Context Reconstruction ──

    def get_active_context(self, max_tokens: int = 8192) -> list[ContextNode]:
        """Select nodes for LLM context via priority BFS from active node.

        Nodes are prioritized by edge_weight * max(node_value, 0.1).
        Root and active node always included first.
        Returns nodes sorted by (depth, creation_order) for coherent presentation.
        """
        if self.active_node_id is None:
            return []

        selected = []
        visited = set()
        total_tokens = 0

        # Priority queue: (-priority, node_id)
        pq = [(-10.0, self.active_node_id)]
        if self.root_id and self.root_id != self.active_node_id:
            pq.append((-9.0, self.root_id))

        while pq and total_tokens < max_tokens:
            neg_priority, nid = heapq.heappop(pq)
            if nid in visited:
                continue
            visited.add(nid)

            node = self.nodes.get(nid)
            if node is None or node.status == NodeStatus.PRUNED:
                continue

            # Skip folded nodes unless they're the root
            if node.status == NodeStatus.FOLDED and nid != self.root_id:
                continue

            if total_tokens + node.token_count > max_tokens:
                continue

            selected.append(node)
            total_tokens += node.token_count

            # Expand neighbors
            for edge in self.edges:
                neighbor = None
                if edge.source == nid and edge.target not in visited:
                    neighbor = edge.target
                elif edge.target == nid and edge.source not in visited:
                    neighbor = edge.source

                if neighbor and neighbor in self.nodes:
                    n = self.nodes[neighbor]
                    if n.status not in (NodeStatus.PRUNED, NodeStatus.FOLDED):
                        priority = edge.weight * max(n.value, 0.1)
                        heapq.heappush(pq, (-priority, neighbor))

        selected.sort(key=lambda n: (n.depth, _seq_key(n.id)))
        return selected

    # ── Serialization ──

    def to_state_text(self, max_nodes_shown: int = 12, max_edges_shown: int = 15) -> str:
        """Compact graph state summary for LLM observation.

        max_nodes_shown / max_edges_shown cap the textual size to keep
        per-turn prompt token budget bounded as the graph grows.
        Always shows root and active node first; other nodes prioritized
        by recency (newest first).
        """
        active = self.active_nodes
        eligible_ids = [node.id for node in active if node.id != self.root_id]
        lines = [
            f"[Graph] {len(active)}/{len(self.nodes)} nodes active, "
            f"{len(self.active_edges)} active edges, ops={self.operation_count}, "
            f"focus=[{self.active_node_id}]",
            "  Eligible graph-tool node IDs: "
            + (", ".join(eligible_ids[-max_nodes_shown:]) if eligible_ids else "(none)"),
        ]

        visible = [(nid, n) for nid, n in self.nodes.items() if n.is_active()]
        # Priority order: root → active → newest first
        def sort_key(item):
            nid, node = item
            if nid == self.root_id:
                return (0, 0)
            if nid == self.active_node_id:
                return (1, 0)
            return (2, -_seq_key(nid))  # newest first
        visible.sort(key=sort_key)

        truncated_nodes = max(0, len(visible) - max_nodes_shown)
        for nid, node in visible[:max_nodes_shown]:
            status_mark = "*" if nid == self.active_node_id else ""
            preview = node.content[:60].replace("\n", " ")
            if len(node.content) > 60:
                preview += "..."
            lines.append(
                f"  [{nid}]{status_mark} {node.type.value}: {preview} [{node.token_count}tok]"
            )
        if truncated_nodes > 0:
            lines.append(f"  ... and {truncated_nodes} more nodes (hidden)")

        if self.edges:
            edge_strs = []
            for e in self.edges:
                src = self.nodes.get(e.source)
                tgt = self.nodes.get(e.target)
                if src and tgt and src.is_active() and tgt.is_active():
                    edge_strs.append(f"{e.source}->{e.target}({e.relation.value})")
            if edge_strs:
                lines.append("  Edges: " + ", ".join(edge_strs[:max_edges_shown]))
                if len(edge_strs) > max_edges_shown:
                    lines.append(f"  ... and {len(edge_strs) - max_edges_shown} more edges")

        return "\n".join(lines)

    def to_context_text(self, max_tokens: int = 2048) -> str:
        """Build detailed context text from active subgraph for LLM prompt.

        Default cap reduced from 8192 to 2048 to keep total per-turn injection
        bounded (state_text + context_text + base prompt must fit within
        the model's effective forward budget).
        """
        nodes = self.get_active_context(max_tokens)
        if not nodes:
            return ""

        parts = []
        for node in nodes:
            marker = " [FOCUS]" if node.id == self.active_node_id else ""
            parts.append(f"--- [{node.id}] {node.type.value}{marker} ---")
            parts.append(node.content)
            parts.append("")

        return "\n".join(parts)

    def retrieve_context(
        self,
        query: str,
        summary_budget: int = 768,
        evidence_budget: int = 1280,
        max_summaries: int = 5,
        max_evidence: int = 4,
        exclude_node_ids: Optional[set[str]] = None,
    ) -> str:
        """Build a bounded, query-conditioned working-memory view.

        Summaries remain in the active parent graph, while raw observations
        may live either on the parent or in detached child archives. Retrieval
        first selects semantic summaries, then gives their linked archives a
        score boost before choosing raw evidence. The archive never becomes
        part of ordinary graph serialization, so prompt size is independent
        of total stored evidence.
        """
        exclude_node_ids = exclude_node_ids or set()
        query_terms = _lexical_terms(query)
        root_terms = _lexical_terms(
            self.nodes[self.root_id].content
            if self.root_id and self.root_id in self.nodes
            else ""
        )

        def relevance(
            content: str,
            metadata: Optional[dict] = None,
            recency: int = 0,
        ) -> float:
            terms = _lexical_terms(content)
            metadata_terms = _lexical_terms(
                " ".join(
                    str(value)
                    for key, value in (metadata or {}).items()
                    if key != "raw_content"
                )
            )
            searchable = terms | metadata_terms
            current_overlap = len(searchable & query_terms) / max(
                len(query_terms), 1
            )
            root_overlap = len(searchable & root_terms) / max(len(root_terms), 1)
            # Recency only breaks near-ties; relevance remains query-driven.
            return 2.0 * current_overlap + 0.5 * root_overlap + recency * 1e-6

        summary_candidates = []
        for node in self.nodes.values():
            if (
                node.id in exclude_node_ids
                or node.type != NodeType.SUMMARY
                or node.status == NodeStatus.PRUNED
            ):
                continue
            score = relevance(node.content, node.metadata, _seq_key(node.id))
            score += max(node.value, 0.0) * 0.1
            summary_candidates.append((score, _seq_key(node.id), node))
        summary_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_summaries = [
            item[2] for item in summary_candidates[:max_summaries]
        ]
        selected_archive_ids = {
            node.metadata.get("archive_id")
            for node in selected_summaries
            if node.metadata.get("archive_id")
        }

        evidence_candidates = []
        for node in self.nodes.values():
            if (
                node.id in exclude_node_ids
                or node.type not in (NodeType.OBSERVATION, NodeType.ACTION)
            ):
                continue
            raw = node.metadata.get("raw_content", node.content)
            score = relevance(raw, node.metadata, _seq_key(node.id))
            evidence_candidates.append(
                (score, _seq_key(node.id), node.id, raw, node.metadata)
            )

        archive_sequence = len(self.nodes)
        for archive in self.archives.values():
            linked_bonus = 0.75 if archive.id in selected_archive_ids else 0.0
            for item in archive.evidence:
                if item.id in exclude_node_ids:
                    continue
                score = relevance(
                    item.content,
                    item.metadata,
                    archive_sequence + item.sequence,
                )
                evidence_candidates.append(
                    (
                        score + linked_bonus,
                        archive_sequence + item.sequence,
                        item.id,
                        item.content,
                        item.metadata,
                    )
                )
            archive_sequence += len(archive.evidence) + 1

        evidence_candidates.sort(
            key=lambda item: (item[0], item[1]), reverse=True
        )
        selected_evidence = []
        seen_content = set()
        for candidate in evidence_candidates:
            normalized = " ".join(candidate[3].lower().split())
            if not normalized or normalized in seen_content:
                continue
            seen_content.add(normalized)
            selected_evidence.append(candidate)
            if len(selected_evidence) >= max_evidence:
                break

        def format_blocks(items, budget, block_builder):
            blocks = []
            remaining = max(0, int(budget))
            for item in items:
                header, content = block_builder(item)
                header_cost = self._count_tokens(header)
                if remaining <= header_cost:
                    break
                content_budget = remaining - header_cost
                fitted = self._truncate_tokens(content, content_budget)
                if not fitted:
                    continue
                block = f"{header}\n{fitted}"
                # Tokenization is not perfectly additive around the newline;
                # tighten the content until the complete block fits exactly.
                while fitted and self._count_tokens(block) > remaining:
                    overflow = self._count_tokens(block) - remaining
                    content_budget -= max(overflow, 1)
                    fitted = self._truncate_tokens(content, content_budget)
                    block = f"{header}\n{fitted}" if fitted else ""
                if not block:
                    continue
                blocks.append(block)
                remaining -= self._count_tokens(block)
                if remaining <= 0:
                    break
            return blocks

        summary_blocks = format_blocks(
            selected_summaries,
            summary_budget,
            lambda node: (f"[Summary {node.id}]", node.content),
        )
        evidence_blocks = format_blocks(
            selected_evidence,
            evidence_budget,
            lambda item: (
                f"[Evidence {item[2]} tool={item[4].get('tool', 'unknown')}]",
                item[3],
            ),
        )

        self.last_retrieval_stats = {
            "n_summaries": len(summary_blocks),
            "n_evidence": len(evidence_blocks),
            "summary_tokens": sum(self._count_tokens(x) for x in summary_blocks),
            "evidence_tokens": sum(self._count_tokens(x) for x in evidence_blocks),
            "n_archives": len(self.archives),
        }
        sections = []
        if summary_blocks:
            sections.append("Relevant consolidated memory:\n" + "\n\n".join(summary_blocks))
        if evidence_blocks:
            sections.append("Recoverable source evidence:\n" + "\n\n".join(evidence_blocks))
        return "\n\n".join(sections)

    @property
    def active_nodes(self) -> list[ContextNode]:
        return [n for n in self.nodes.values() if n.is_active()]

    @property
    def active_edges(self) -> list[ContextEdge]:
        return [
            edge for edge in self.edges
            if edge.source in self.nodes
            and edge.target in self.nodes
            and self.nodes[edge.source].is_active()
            and self.nodes[edge.target].is_active()
        ]

    def is_saturated(
        self,
        max_active: int = 8,
        max_edges: int = 15,
        max_idle_turns: int = 3,
    ) -> bool:
        """Whether the graph is dense enough that <pass> is a valid consolidation
        response. Used to gate the negative reward at forced-consolidation
        checkpoints in graph_agent_isolated. Defaults are tuned from BC smoke
        data (val mean n_active=4.3, n_edges=8.8) -- thresholds sit in the
        distribution tail.
        """
        if len(self.active_nodes) > max_active:
            return True
        if len(self.active_edges) > max_edges:
            return True
        if self.turns_since_last_node_add >= max_idle_turns:
            return True
        return False

    @property
    def subtask_nodes(self) -> list[ContextNode]:
        return [
            n
            for n in self.nodes.values()
            if n.type == NodeType.SUBTASK and n.is_active()
        ]

    # ── Automatic Graph Operations (heuristic warm-start) ──

    def auto_prune_low_value(self, max_active: int = 20, min_value: float = -0.5) -> list[str]:
        """Auto-prune observation nodes when graph exceeds max_active.

        Strategy: Remove lowest-value observation nodes first (search results
        that weren't referenced by any edge beyond the auto-temporal one).
        Never prune root, subtask, or summary nodes.
        """
        pruned = []
        active = self.active_nodes
        if len(active) <= max_active:
            return pruned

        # Candidates: observation nodes sorted by value (ascending)
        candidates = sorted(
            [n for n in active if n.type == NodeType.OBSERVATION],
            key=lambda n: (n.value, -_seq_key(n.id)),  # lowest value first, oldest first
        )

        for node in candidates:
            if len(self.active_nodes) <= max_active:
                break
            # Don't prune if node has outgoing edges (was referenced)
            outgoing = [e for e in self.edges if e.source == node.id and e.relation != EdgeRelation.TEMPORAL]
            if not outgoing:
                self.prune(node.id)
                pruned.append(node.id)

        return pruned

    def auto_merge_similar(self, similarity_threshold: int = 3) -> Optional[str]:
        """Auto-merge observation nodes from the same subtask that share keywords.

        Strategy: For each subtask node, if it has >= similarity_threshold
        observation children, merge them into a summary.
        Returns the merged node id or None.
        """
        for node in list(self.nodes.values()):
            if node.type != NodeType.SUBTASK or node.status != NodeStatus.ACTIVE:
                continue

            # Find active observation children
            children = []
            for edge in self.edges:
                if edge.source == node.id:
                    child = self.nodes.get(edge.target)
                    if child and child.type == NodeType.OBSERVATION and child.is_active():
                        children.append(child)

            if len(children) >= similarity_threshold:
                children = children[:6]
                # Auto-merge: create summary from children content
                combined = "\n".join(f"[{c.id}]: {c.content[:200]}" for c in children)
                summary = f"[Auto-merged from {node.id}] {len(children)} findings:\n{combined}"
                child_ids = [c.id for c in children]
                merged_id = self.merge(child_ids, summary)
                if merged_id:
                    return merged_id
        return None

    def auto_connect_semantic(self, keyword_overlap_threshold: int = 3) -> list[tuple[str, str]]:
        """Auto-add semantic edges between nodes sharing significant keyword overlap.

        Strategy: For observation/summary nodes from different subtasks,
        if they share >= threshold content words, add a semantic edge.
        Only checks active nodes. Simple word-overlap heuristic.
        """
        added = []
        active_obs = [
            n for n in self.nodes.values()
            if n.is_active() and n.type in (NodeType.OBSERVATION, NodeType.SUMMARY)
        ]

        # Build word sets per node (simple tokenization)
        word_sets = {}
        for n in active_obs:
            words = set(n.content.lower().split())
            # Filter short/common words
            words = {w for w in words if len(w) > 4}
            word_sets[n.id] = words

        # Find parent subtask for each node
        def get_parent_subtask(nid):
            for e in self.edges:
                if e.target == nid:
                    parent = self.nodes.get(e.source)
                    if parent and parent.type == NodeType.SUBTASK:
                        return parent.id
            return None

        # Check pairs from different subtasks
        checked = set()
        for i, n1 in enumerate(active_obs):
            p1 = get_parent_subtask(n1.id)
            for n2 in active_obs[i + 1:]:
                p2 = get_parent_subtask(n2.id)
                if p1 == p2:
                    continue  # Same subtask, skip
                pair = (min(n1.id, n2.id), max(n1.id, n2.id))
                if pair in checked:
                    continue
                checked.add(pair)

                overlap = len(word_sets.get(n1.id, set()) & word_sets.get(n2.id, set()))
                if overlap >= keyword_overlap_threshold:
                    # Check if edge already exists
                    exists = any(
                        e.source == n1.id and e.target == n2.id and e.relation == EdgeRelation.SEMANTIC
                        for e in self.edges
                    )
                    if not exists:
                        self.add_edge(n1.id, n2.id, EdgeRelation.SEMANTIC, weight=min(overlap / 10, 1.0))
                        added.append((n1.id, n2.id))

        return added

    # ── Reward Computation ──

    def _compute_branch_uniqueness(self) -> float:
        """Token-level Jaccard uniqueness across active SUMMARY nodes.

        For each active summary node (typically created by collapsing a
        finished branch), compute the fraction of its tokens that do not
        appear in any other active summary. Sum across summaries, scaled,
        bounded to 0.3.

        Returns 0 if fewer than 2 active summaries exist (uniqueness is
        only meaningful relative to siblings).

        Cheap: O(n_summaries * n_tokens_per_summary).
        """
        summaries = [
            n for n in self.nodes.values()
            if n.type == NodeType.SUMMARY and n.is_active()
        ]
        if len(summaries) < 2:
            return 0.0
        token_sets = {n.id: set(n.content.lower().split()) for n in summaries}
        scores = []
        for nid, t in token_sets.items():
            if not t:
                continue
            others = set().union(
                *(v for k, v in token_sets.items() if k != nid)
            )
            scores.append(len(t - others) / len(t))
        if not scores:
            return 0.0
        return min(sum(scores) * 0.05, 0.3)

    def compute_graph_reward(
        self,
        task_reward: float,
        lambda_compact: float = 0.1,
        lambda_cost: float = 0.02,
        uniqueness_weight: float = 0.0,
    ) -> dict:
        """Compute graph-aware reward conditioned on task success.

        Design principles:
        1. Positive graph shaping is ONLY applied when task succeeds.
           Failed tasks can only receive an invalid-operation penalty.
        2. Compactness: reward fewer active nodes (efficient context management).
        3. Structural quality: reward well-connected graphs (meaningful cross-branch links).
        4. Merge utility: reward merges that compress without losing task performance.
        5. Valid-op cost is outcome-gated; invalid calls are always penalized.

        Returns dict with all reward components for logging.
        """
        n_active = len(self.active_nodes)
        n_total = max(len(self.nodes), 1)
        n_folded = sum(1 for n in self.nodes.values() if n.status == NodeStatus.FOLDED)
        n_pruned = sum(1 for n in self.nodes.values() if n.status == NodeStatus.PRUNED)
        active_edges = self.active_edges
        n_edges_active = len(active_edges)
        n_summaries = sum(1 for n in self.nodes.values() if n.type == NodeType.SUMMARY and n.is_active())
        n_cross_edges = sum(
            1
            for e in active_edges
            if e.relation in (EdgeRelation.SEMANTIC, EdgeRelation.CAUSAL)
        )

        # ── Graph shaping (strictly task-gated) ──
        # Structural diagnostics remain visible regardless of outcome, but only
        # successful tasks can turn them into positive shaping.
        structural = 0.0
        if n_active > 1:
            structural = min(n_cross_edges / max(n_active - 1, 1), 1.0)
        usage_bonus = 0.0

        # Branch uniqueness (Improvement #1, gated by uniqueness_weight; default 0)
        # Rewards branches that surface info not already in sibling summaries —
        # incentivizes branch_success to decouple from task_reward (in v3 we
        # observed branch_success ~= task_reward, meaning branches did not add
        # independent value beyond the main agent's final answer).
        uniqueness_raw = self._compute_branch_uniqueness()
        uniqueness_bonus = uniqueness_weight * uniqueness_raw

        if task_reward > 0:
            # Task succeeded: full shaping including outcome-derived terms
            # Compactness: ratio of compression (folded + pruned) / total
            compression_ratio = (n_folded + n_pruned) / n_total if n_total > 1 else 0.0
            compactness = compression_ratio  # [0, 1], higher is better

            # Merge utility: successful task + merges = agent compressed effectively
            merge_bonus = min(n_summaries * 0.1, 0.3)

            # Prune utility: successful task + prunes = agent discarded correctly
            prune_bonus = min(n_pruned * 0.05, 0.15)

            raw_graph_shaping = (
                lambda_compact * (compactness + structural + merge_bonus + prune_bonus)
                + uniqueness_bonus
            )
            graph_shaping = min(max(raw_graph_shaping, 0.0), 0.1)
        else:
            # Failed answers cannot earn positive graph reward. Tool mechanics
            # are learned through SFT and invalid-call penalties instead.
            compactness = 0.0
            merge_bonus = 0.0
            prune_bonus = 0.0
            graph_shaping = 0.0

        # ── Cost penalty (only counts LLM-initiated graph ops) ──
        cost = self.explicit_op_count
        cost_penalty = lambda_cost * cost if task_reward > 0 else 0.0
        invalid_op_penalty = 0.01 * self.invalid_op_count

        # Context bloat is now handled by auto-merge (graph_agent_isolated.py),
        # not reward penalty. This avoids punishing the agent for something
        # it can't control (4B can't learn merge/prune from scratch).
        bloat_penalty = 0.0

        r_graph = task_reward + graph_shaping - cost_penalty - invalid_op_penalty

        return {
            "task_reward": task_reward,
            "graph_reward": r_graph,
            "graph_shaping": graph_shaping,
            "compactness": compactness,
            "structural": structural,
            "merge_bonus": merge_bonus,
            "prune_bonus": prune_bonus,
            "usage_bonus": usage_bonus,
            "uniqueness_bonus": uniqueness_bonus,
            "uniqueness_raw": uniqueness_raw,
            "cost_penalty": cost_penalty,
            "invalid_op_penalty": invalid_op_penalty,
            "bloat_penalty": bloat_penalty,
            "operation_cost": cost,
            "total_ops": self.operation_count,
            "explicit_ops": self.explicit_op_count,
            "graph_op_attempts": self.graph_op_attempt_count,
            "invalid_ops": self.invalid_op_count,
            "invalid_op_rate": (
                self.invalid_op_count / self.graph_op_attempt_count
                if self.graph_op_attempt_count else 0.0
            ),
            "n_active": n_active,
            "n_total": len(self.nodes),
            "n_folded": n_folded,
            "n_pruned": n_pruned,
            "n_edges": n_edges_active,
            "n_cross_edges": n_cross_edges,
            "n_summaries": n_summaries,
        }
