"""Structured, replay-auditable ContextGraph trajectory traces."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from .context_graph import ContextGraph


TRACE_SCHEMA_VERSION = "contextgraph.trace.v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_hash(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest()


def validate_graph_snapshot(snapshot: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(snapshot, dict):
        return ["graph snapshot is not an object"]
    nodes = snapshot.get("nodes")
    edges = snapshot.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return ["graph snapshot requires nodes and edges lists"]
    node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    if len(node_ids) != len(nodes) or any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
        errors.append("every node requires a non-empty string id")
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node ids")
    known = set(node_ids)
    root_id = snapshot.get("root_id")
    active_id = snapshot.get("active_node_id")
    if root_id is not None and root_id not in known:
        errors.append("root_id does not reference a node")
    if active_id is not None and active_id not in known:
        errors.append("active_node_id does not reference a node")
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edge {index} is not an object")
            continue
        if edge.get("source") not in known or edge.get("target") not in known:
            errors.append(f"edge {index} references an unknown endpoint")
    return errors


class GraphTraceRecorder:
    """Capture complete before/after graph states for each parent-graph mutation."""

    def __init__(self, graph: ContextGraph):
        initial = graph.to_dict(include_archives=False)
        self.initial_graph = deepcopy(initial)
        self.initial_hash = snapshot_hash(initial)
        self.events: list[dict[str, Any]] = []
        self._rendered_by_hash = {self.initial_hash: graph.to_state_text()}

    def capture(self, graph: ContextGraph) -> dict[str, Any]:
        snapshot = deepcopy(graph.to_dict(include_archives=False))
        self._rendered_by_hash[snapshot_hash(snapshot)] = graph.to_state_text()
        return snapshot

    def record(
        self,
        graph: ContextGraph,
        before: dict[str, Any],
        *,
        turn_id: int,
        source: str,
        op: str,
        args: dict[str, Any] | None = None,
        success: bool = True,
        error: str | None = None,
        assistant_content: str | None = None,
    ) -> dict[str, Any]:
        after = self.capture(graph)
        before_nodes = {node["id"]: node for node in before.get("nodes", [])}
        after_nodes = {node["id"]: node for node in after.get("nodes", [])}
        before_edges = {canonical_json(edge) for edge in before.get("edges", [])}
        after_edges = {canonical_json(edge) for edge in after.get("edges", [])}
        event = {
            "seq": len(self.events),
            "turn_id": int(turn_id),
            "source": source,
            "op": op,
            "args": ContextGraph._json_safe(args or {}),
            "success": bool(success),
            "error": error,
            "assistant_content_sha256": (
                hashlib.sha256(assistant_content.encode("utf-8")).hexdigest()
                if assistant_content is not None else None
            ),
            "assistant_content_preview": (
                assistant_content[-512:] if assistant_content is not None else None
            ),
            "before_hash": snapshot_hash(before),
            "after_hash": snapshot_hash(after),
            "before_state": deepcopy(before),
            "after_state": after,
            "rendered_before": self._rendered_by_hash.get(
                snapshot_hash(before), _render_snapshot(before)
            ),
            "rendered_after": graph.to_state_text(),
            "created_node_ids": sorted(set(after_nodes) - set(before_nodes)),
            "removed_node_ids": sorted(set(before_nodes) - set(after_nodes)),
            "status_changes": [
                {
                    "node_id": node_id,
                    "before": before_nodes[node_id].get("status"),
                    "after": after_nodes[node_id].get("status"),
                }
                for node_id in sorted(set(before_nodes) & set(after_nodes))
                if before_nodes[node_id].get("status") != after_nodes[node_id].get("status")
            ],
            "added_edges": [json.loads(edge) for edge in sorted(after_edges - before_edges)],
            "removed_edges": [json.loads(edge) for edge in sorted(before_edges - after_edges)],
            "active_node_before": before.get("active_node_id"),
            "active_node_after": after.get("active_node_id"),
        }
        event["rendered_before_hash"] = hashlib.sha256(
            event["rendered_before"].encode("utf-8")
        ).hexdigest()
        event["rendered_after_hash"] = hashlib.sha256(
            event["rendered_after"].encode("utf-8")
        ).hexdigest()
        self.events.append(event)
        return event

    def finalize(self, graph: ContextGraph) -> dict[str, Any]:
        final_graph = self.capture(graph)
        archive_snapshot = graph.to_dict(include_archives=True).get("archives", [])
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "initial_graph": deepcopy(self.initial_graph),
            "initial_hash": self.initial_hash,
            "events": deepcopy(self.events),
            "final_graph": final_graph,
            "final_hash": snapshot_hash(final_graph),
            # Archives may contain long raw branch evidence. Store them once,
            # not in every event snapshot.
            "archives": archive_snapshot,
        }


def _render_snapshot(snapshot: dict[str, Any]) -> str:
    """Compact deterministic representation for auditing what IDs were visible."""
    active_nodes = [node for node in snapshot.get("nodes", []) if node.get("status") == "active"]
    ids = ", ".join(node.get("id", "") for node in active_nodes)
    return (
        f"root={snapshot.get('root_id')} active={snapshot.get('active_node_id')} "
        f"nodes={len(snapshot.get('nodes', []))} edges={len(snapshot.get('edges', []))} "
        f"eligible=[{ids}]"
    )


def validate_graph_trace(trace: Any) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate snapshot hashes, event continuity, and operation effects."""
    errors: list[str] = []
    metrics = {
        "model_op_attempts": 0,
        "valid_model_ops": 0,
        "explicit_valid_model_ops": 0,
        "structural_model_ops": 0,
        "productive_model_ops": 0,
        "redundant_model_ops": 0,
        "semantic_model_op_errors": 0,
        "quality_score": 0.0,
    }
    if not isinstance(trace, dict):
        return False, ["graph trace is not an object"], metrics
    if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
        errors.append("unsupported graph trace schema")
    initial = trace.get("initial_graph")
    final = trace.get("final_graph")
    events = trace.get("events")
    errors.extend(f"initial: {error}" for error in validate_graph_snapshot(initial))
    errors.extend(f"final: {error}" for error in validate_graph_snapshot(final))
    if isinstance(initial, dict) and trace.get("initial_hash") != snapshot_hash(initial):
        errors.append("initial graph hash mismatch")
    if isinstance(final, dict) and trace.get("final_hash") != snapshot_hash(final):
        errors.append("final graph hash mismatch")
    if not isinstance(events, list):
        return False, errors + ["events is not a list"], metrics

    expected_hash = trace.get("initial_hash")
    explicit_ops = {"merge", "add_edge", "select", "prune"}
    structural_ops = {"merge", "add_edge", "prune"}
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"event {index} is not an object")
            continue
        before = event.get("before_state")
        after = event.get("after_state")
        errors.extend(f"event {index} before: {error}" for error in validate_graph_snapshot(before))
        errors.extend(f"event {index} after: {error}" for error in validate_graph_snapshot(after))
        if isinstance(before, dict) and event.get("before_hash") != snapshot_hash(before):
            errors.append(f"event {index} before hash mismatch")
        if isinstance(after, dict) and event.get("after_hash") != snapshot_hash(after):
            errors.append(f"event {index} after hash mismatch")
        if event.get("before_hash") != expected_hash:
            errors.append(f"event {index} is not continuous with the previous state")
        for side in ("before", "after"):
            rendered = event.get(f"rendered_{side}")
            rendered_hash = event.get(f"rendered_{side}_hash")
            if not isinstance(rendered, str) or rendered_hash != hashlib.sha256(rendered.encode("utf-8")).hexdigest():
                errors.append(f"event {index} rendered {side} hash mismatch")
        expected_hash = event.get("after_hash")

        if event.get("source") != "model":
            continue
        metrics["model_op_attempts"] += 1
        success = bool(event.get("success"))
        op = event.get("op")
        if success:
            metrics["valid_model_ops"] += 1
        if success and op in explicit_ops:
            metrics["explicit_valid_model_ops"] += 1
        if success and op in structural_ops:
            metrics["structural_model_ops"] += 1
        semantic_errors = _model_operation_errors(event) if success else []
        if semantic_errors:
            metrics["semantic_model_op_errors"] += len(semantic_errors)
            errors.extend(
                f"event {index} semantic: {error}" for error in semantic_errors
            )
        productive = success and _has_expected_effect(event)
        if productive:
            metrics["productive_model_ops"] += 1
        elif success and op != "pass":
            metrics["redundant_model_ops"] += 1

    if expected_hash != trace.get("final_hash"):
        errors.append("final graph is not continuous with the last event")
    attempts = metrics["model_op_attempts"]
    metrics["quality_score"] = (
        metrics["productive_model_ops"] / attempts if attempts else 0.0
    )
    return not errors, errors, metrics


def _has_expected_effect(event: dict[str, Any]) -> bool:
    op = event.get("op")
    before = event.get("before_state") or {}
    after = event.get("after_state") or {}
    args = event.get("args") or {}
    before_nodes = {node.get("id"): node for node in before.get("nodes", [])}
    after_nodes = {node.get("id"): node for node in after.get("nodes", [])}
    if op == "pass":
        return bool(event.get("success")) and event.get("before_hash") == event.get("after_hash")
    if op == "select":
        target = args.get("node_id")
        return after.get("active_node_id") == target and before.get("active_node_id") != target
    if op == "prune":
        target = args.get("node_id")
        return before_nodes.get(target, {}).get("status") == "active" and after_nodes.get(target, {}).get("status") == "pruned"
    if op == "add_edge":
        return bool(event.get("added_edges"))
    if op == "merge":
        return bool(event.get("created_node_ids")) and bool(event.get("status_changes"))
    if op == "branch":
        created = [after_nodes[node_id] for node_id in event.get("created_node_ids", []) if node_id in after_nodes]
        created_types = {node.get("type") for node in created}
        return "subtask" in created_types and "summary" in created_types
    if event.get("source") in {"environment", "heuristic", "system"}:
        return event.get("before_hash") != event.get("after_hash")
    return event.get("before_hash") != event.get("after_hash")


def _model_operation_errors(event: dict[str, Any]) -> list[str]:
    """Validate that successful calls were meaningful in their pre-state."""
    op = event.get("op")
    args = event.get("args") or {}
    before = event.get("before_state") or {}
    after = event.get("after_state") or {}
    before_nodes = {node.get("id"): node for node in before.get("nodes", [])}
    after_nodes = {node.get("id"): node for node in after.get("nodes", [])}
    active_before = {
        node_id for node_id, node in before_nodes.items()
        if node.get("status") == "active"
    }
    errors: list[str] = []
    if op == "add_edge":
        source, target = args.get("source"), args.get("target")
        relation = args.get("relation", "semantic")
        if source == target or source not in active_before or target not in active_before:
            errors.append("add_edge endpoints were not distinct active nodes")
        if relation not in {"causal", "semantic", "temporal"}:
            errors.append("add_edge used an unsupported relation")
        if not any(
            edge.get("source") == source
            and edge.get("target") == target
            and edge.get("relation") == relation
            for edge in event.get("added_edges", [])
        ):
            errors.append("add_edge transition does not match its arguments")
    elif op == "merge":
        raw_ids = args.get("node_ids", [])
        node_ids = (
            [item.strip() for item in raw_ids.split(",") if item.strip()]
            if isinstance(raw_ids, str) else list(raw_ids)
        )
        if len(node_ids) < 2 or len(node_ids) != len(set(node_ids)):
            errors.append("merge requires at least two unique source nodes")
        if any(node_id not in active_before or node_id == before.get("root_id") for node_id in node_ids):
            errors.append("merge sources were not eligible active nodes")
        if not str(args.get("summary", "")).strip():
            errors.append("merge summary is empty")
        created = [
            after_nodes[node_id]
            for node_id in event.get("created_node_ids", [])
            if node_id in after_nodes and after_nodes[node_id].get("type") == "summary"
        ]
        if len(created) != 1 or created[0].get("metadata", {}).get("merged_from") != node_ids:
            errors.append("merge summary provenance does not match its sources")
    elif op == "prune":
        target = args.get("node_id")
        if target not in active_before or target == before.get("root_id"):
            errors.append("prune target was not an eligible active node")
        if after_nodes.get(target, {}).get("status") != "pruned":
            errors.append("prune did not mark its target pruned")
    elif op == "select":
        if args.get("node_id") not in active_before:
            errors.append("select target was not active in the pre-state")
    elif op == "branch":
        if not str(args.get("prompt", "")).strip():
            errors.append("branch prompt is empty")
    elif op == "pass":
        if event.get("before_hash") != event.get("after_hash"):
            errors.append("pass unexpectedly mutated the graph")
    return errors
