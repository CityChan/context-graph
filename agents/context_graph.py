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
from typing import Optional


_SEQ_RE = re.compile(r'(\d+)$')


def _seq_key(nid: str) -> int:
    """Extract trailing integer from a node ID for stable sorting.

    Handles both single-prefix IDs ('n5') and multi-char prefix IDs ('b0_12')
    used by spawned subgraphs.
    """
    m = _SEQ_RE.search(nid)
    return int(m.group(1)) if m else 0


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
        self.tokenizer = tokenizer
        self._node_counter = 0
        # Hierarchical fields (used by isolated graph_agent variant)
        self.parent: Optional[ContextGraph] = parent
        self.namespace_prefix: str = namespace_prefix

    def _next_id(self, prefix: Optional[str] = None) -> str:
        self._node_counter += 1
        return f"{prefix or self.namespace_prefix}{self._node_counter}"

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

    def collapse_child(self, parent_node_id: str) -> Optional[dict]:
        """Drop the subgraph attached to a SUBTASK node, returning summary stats.

        Called after a branch returns and its findings have been folded into
        a SUMMARY node on the parent. Releases the child graph's memory so
        the parent's training trajectory does not carry the child's nodes.
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
        node.child_graph = None
        return stats

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text.split())

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
        """Add an edge between two existing nodes. Updates weight if duplicate."""
        if source not in self.nodes or target not in self.nodes:
            return False
        for e in self.edges:
            if e.source == source and e.target == target and e.relation == relation:
                e.weight = weight
                return True
        self.edges.append(ContextEdge(source, target, relation, weight))
        self.operation_count += 1
        return True

    def merge(self, node_ids: list[str], summary: str) -> Optional[str]:
        """Merge multiple nodes into a single summary node.

        Source nodes are marked as FOLDED. The new summary node inherits
        incoming edges from the merged nodes.
        """
        valid_ids = [nid for nid in node_ids if nid in self.nodes and nid != self.root_id]
        if len(valid_ids) < 2:
            return None

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
            self.nodes[nid].status = NodeStatus.FOLDED

        # Inherit incoming edges from merged nodes
        for edge in list(self.edges):
            if edge.target in valid_ids and edge.source not in valid_ids:
                self.add_edge(edge.source, merged_id, edge.relation, edge.weight)

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
        if self.nodes[node_id].status == NodeStatus.PRUNED:
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
        if node_id not in self.nodes or node_id == self.root_id:
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
        lines = [
            f"[Graph] {len(active)}/{len(self.nodes)} nodes active, "
            f"{len(self.edges)} edges, ops={self.operation_count}, "
            f"focus=[{self.active_node_id}]"
        ]

        visible = [(nid, n) for nid, n in self.nodes.items() if n.status != NodeStatus.PRUNED]
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
            fold_mark = "(folded)" if node.status == NodeStatus.FOLDED else ""
            preview = node.content[:60].replace("\n", " ")
            if len(node.content) > 60:
                preview += "..."
            lines.append(
                f"  [{nid}]{status_mark} {node.type.value} {fold_mark}: {preview} [{node.token_count}tok]"
            )
        if truncated_nodes > 0:
            lines.append(f"  ... and {truncated_nodes} more nodes (hidden)")

        if self.edges:
            edge_strs = []
            for e in self.edges:
                src = self.nodes.get(e.source)
                tgt = self.nodes.get(e.target)
                if src and tgt and src.status != NodeStatus.PRUNED and tgt.status != NodeStatus.PRUNED:
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

    @property
    def active_nodes(self) -> list[ContextNode]:
        return [n for n in self.nodes.values() if n.is_active()]

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

    def compute_graph_reward(
        self,
        task_reward: float,
        lambda_compact: float = 0.1,
        lambda_cost: float = 0.005,
    ) -> dict:
        """Compute graph-aware reward conditioned on task success.

        Design principles:
        1. Graph shaping rewards are ONLY applied when task succeeds (task_reward > 0).
           Otherwise graph ops should not be rewarded/penalized (avoid credit mis-assignment).
        2. Compactness: reward fewer active nodes (efficient context management).
        3. Structural quality: reward well-connected graphs (meaningful cross-branch links).
        4. Merge utility: reward merges that compress without losing task performance.
        5. Cost: always penalize excessive operations (prevents reward hacking).

        Returns dict with all reward components for logging.
        """
        n_active = len(self.active_nodes)
        n_total = max(len(self.nodes), 1)
        n_folded = sum(1 for n in self.nodes.values() if n.status == NodeStatus.FOLDED)
        n_pruned = sum(1 for n in self.nodes.values() if n.status == NodeStatus.PRUNED)
        n_edges_active = sum(
            1
            for e in self.edges
            if self.nodes.get(e.source, ContextNode("", NodeType.QUERY, "")).status != NodeStatus.PRUNED
            and self.nodes.get(e.target, ContextNode("", NodeType.QUERY, "")).status != NodeStatus.PRUNED
        )
        n_summaries = sum(1 for n in self.nodes.values() if n.type == NodeType.SUMMARY and n.is_active())
        n_cross_edges = sum(
            1
            for e in self.edges
            if e.relation in (EdgeRelation.SEMANTIC, EdgeRelation.CAUSAL)
            and self.nodes.get(e.source, ContextNode("", NodeType.QUERY, "")).status != NodeStatus.PRUNED
        )

        # ── Graph shaping (only when task succeeds) ──
        if task_reward > 0:
            # Compactness: ratio of compression (folded + pruned) / total
            # Higher = more compressed = better context management
            compression_ratio = (n_folded + n_pruned) / n_total if n_total > 1 else 0.0
            compactness = compression_ratio  # [0, 1], higher is better

            # Structural quality: cross-branch connections show synthesis
            structural = 0.0
            if n_active > 1:
                structural = min(n_cross_edges / n_active, 1.0)  # [0, 1]

            # Merge utility: successful task + merges = agent compressed effectively
            merge_bonus = min(n_summaries * 0.1, 0.3)

            # Prune utility: successful task + prunes = agent discarded correctly
            prune_bonus = min(n_pruned * 0.05, 0.15)

            graph_shaping = lambda_compact * (compactness + structural + merge_bonus + prune_bonus)
        else:
            # Task failed: no graph shaping (don't reward/penalize graph structure)
            compactness = 0.0
            structural = 0.0
            merge_bonus = 0.0
            prune_bonus = 0.0
            graph_shaping = 0.0

        # ── Cost penalty (only counts LLM-initiated graph ops) ──
        # Uses explicit_op_count (merge/prune/select/add_edge tool calls) instead
        # of operation_count (which also includes auto-heuristic ops and observation
        # node creation). This ensures that an agent doing zero graph ops gets
        # zero cost_penalty, making it degenerate cleanly to FoldAgent's reward.
        cost = self.explicit_op_count
        cost_penalty = lambda_cost * cost

        r_graph = task_reward + graph_shaping - cost_penalty

        return {
            "task_reward": task_reward,
            "graph_reward": r_graph,
            "graph_shaping": graph_shaping,
            "compactness": compactness,
            "structural": structural,
            "merge_bonus": merge_bonus,
            "prune_bonus": prune_bonus,
            "cost_penalty": cost_penalty,
            "operation_cost": cost,
            "total_ops": self.operation_count,
            "explicit_ops": self.explicit_op_count,
            "n_active": n_active,
            "n_total": len(self.nodes),
            "n_folded": n_folded,
            "n_pruned": n_pruned,
            "n_edges": n_edges_active,
            "n_cross_edges": n_cross_edges,
            "n_summaries": n_summaries,
        }
