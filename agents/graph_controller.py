"""Controller-owned, structured ContextGraph action decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .context_graph import ContextGraph


class GraphControllerError(ValueError):
    """Raised when a constrained graph decision cannot be safely executed."""


def merge_decision_schema(indices: list[int]) -> dict[str, Any]:
    """Build the shared runtime/preflight schema for a merge decision."""
    if len(indices) < 2:
        raise GraphControllerError("fewer than two merge candidates")
    return {
        "type": "object",
        "properties": {
            "candidate_indices": {
                "type": "array",
                "items": {"type": "integer", "enum": indices},
                "minItems": 2,
                "maxItems": min(6, len(indices)),
            },
            "summary": {"type": "string", "minLength": 1},
        },
        "required": ["candidate_indices", "summary"],
        "additionalProperties": False,
    }


def graph_action_schema(
    indices: list[int],
    *,
    allow_pass: bool = False,
) -> dict[str, Any]:
    """Build a constrained schema for the complete graph action space.

    The schema deliberately stays flat: the deployed llguidance backend does
    not reliably support conditional JSON Schema constructs. Action-specific
    arity and field validation therefore remain controller-owned.
    """
    if not indices:
        raise GraphControllerError("no graph action candidates")
    actions = ["prune", "select"]
    if len(indices) >= 2:
        actions = ["merge", "prune", "add_edge", "select"]
    if allow_pass:
        actions.append("pass")
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": actions,
            },
            "candidate_indices": {
                "type": "array",
                "items": {"type": "integer", "enum": indices},
                "minItems": 0,
                "maxItems": min(6, len(indices)),
            },
            "summary": {"type": "string"},
            "relation": {
                "type": "string",
                "enum": ["causal", "semantic", "temporal"],
            },
        },
        "required": ["action", "candidate_indices", "summary", "relation"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class MergeCandidate:
    index: int
    node_id: str
    node_type: str
    content: str


@dataclass(frozen=True)
class MergeCandidateSnapshot:
    graph_hash: str
    candidates: tuple[MergeCandidate, ...]

    def trace_context(self) -> dict[str, Any]:
        return {
            "graph_hash": self.graph_hash,
            "candidates": [
                {
                    "index": candidate.index,
                    "node_id": candidate.node_id,
                    "node_type": candidate.node_type,
                }
                for candidate in self.candidates
            ],
        }


class GraphActionController:
    """Expose semantic graph choices while owning format and node legality.

    The model sees stable candidate indices rather than graph node IDs. A
    dynamic JSON schema allows merge, prune, add_edge, select, and pass. The
    controller validates each action's fields and canonicalizes duplicate merge
    indices because llguidance does not implement JSON Schema ``uniqueItems``.
    The snapshot hash prevents a decision from being applied after the graph has
    changed.
    """

    def __init__(
        self,
        *,
        max_candidates: int = 12,
        preview_chars: int = 360,
        min_completion_tokens: int = 256,
    ):
        if max_candidates < 2:
            raise ValueError("max_candidates must be at least 2")
        if min_completion_tokens < 10:
            raise ValueError("min_completion_tokens must be at least 10")
        self.max_candidates = int(max_candidates)
        self.preview_chars = max(int(preview_chars), 80)
        self.min_completion_tokens = int(min_completion_tokens)

    def has_completion_budget(
        self,
        remaining_tokens: int,
        *,
        protected_tokens: int = 0,
    ) -> bool:
        """Return whether a checkpoint can finish one structured decision."""
        available = max(int(remaining_tokens), 0) - max(
            int(protected_tokens), 0
        )
        return available >= self.min_completion_tokens

    @staticmethod
    def graph_hash(graph: ContextGraph) -> str:
        snapshot = graph.to_dict(include_archives=False)
        canonical = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def snapshot(self, graph: ContextGraph) -> MergeCandidateSnapshot:
        eligible = [
            node for node in graph.active_nodes
            if node.id != graph.root_id
        ]
        # The rendered graph already prioritizes recent working evidence. Keep
        # the same bounded behavior here so a checkpoint cannot inflate the
        # prompt as an episode grows.
        eligible = eligible[-self.max_candidates:]
        candidates = tuple(
            MergeCandidate(
                index=index,
                node_id=node.id,
                node_type=node.type.value,
                content=" ".join(str(node.content).split())[: self.preview_chars],
            )
            for index, node in enumerate(eligible)
        )
        return MergeCandidateSnapshot(
            graph_hash=self.graph_hash(graph),
            candidates=candidates,
        )

    def merge_schema(self, snapshot: MergeCandidateSnapshot) -> dict[str, Any]:
        indices = [candidate.index for candidate in snapshot.candidates]
        return merge_decision_schema(indices)

    def action_schema(
        self,
        snapshot: MergeCandidateSnapshot,
        *,
        allow_pass: bool = False,
    ) -> dict[str, Any]:
        indices = [candidate.index for candidate in snapshot.candidates]
        return graph_action_schema(indices, allow_pass=allow_pass)

    def merge_prompt(self, snapshot: MergeCandidateSnapshot, *, turn_id: int) -> str:
        if len(snapshot.candidates) < 2:
            raise GraphControllerError("fewer than two merge candidates")
        candidate_lines = "\n".join(
            f"  {candidate.index}: [{candidate.node_type}] {candidate.content}"
            for candidate in snapshot.candidates
        )
        return (
            f"[GRAPH MERGE MODE turn={int(turn_id)}]\n"
            "The controller has frozen the legal evidence candidates below. "
            "Choose 2 to 6 distinct candidate indices whose evidence should be "
            "consolidated, and write one meaningful summary covering every "
            "selected item. Return only the JSON object required by the response "
            "schema. Do not emit XML or an environment action.\n"
            f"Candidates:\n{candidate_lines}"
        )

    def action_prompt(
        self,
        snapshot: MergeCandidateSnapshot,
        *,
        turn_id: int,
        allow_pass: bool,
    ) -> str:
        if not snapshot.candidates:
            raise GraphControllerError("no graph action candidates")
        candidate_lines = "\n".join(
            f"  {candidate.index}: [{candidate.node_type}] {candidate.content}"
            for candidate in snapshot.candidates
        )
        pass_rule = (
            "pass is currently legal because the graph is saturated."
            if allow_pass
            else "pass is currently illegal because the graph is not saturated."
        )
        return (
            f"[GRAPH ACTION MODE turn={int(turn_id)}]\n"
            "The controller has frozen the legal evidence candidates below. "
            "Choose the single graph action that best preserves task-relevant "
            "evidence; do not merge or connect unrelated evidence. Actions: "
            "merge uses 2-6 indices and a meaningful summary; prune uses one "
            "irrelevant/redundant index; add_edge uses exactly two ordered "
            "indices (source, target) and a causal/semantic/temporal relation; "
            "select uses one index to shift focus; pass uses no indices. "
            f"{pass_rule} For fields unused by an action, emit an empty summary "
            "and relation=semantic. Return only the JSON object required by the "
            "response schema. Do not emit XML or an environment action.\n"
            f"Candidates:\n{candidate_lines}"
        )

    def structured_outputs(
        self,
        snapshot: MergeCandidateSnapshot,
        *,
        allow_pass: bool = False,
    ) -> dict[str, Any]:
        return {
            "json": self.action_schema(snapshot, allow_pass=allow_pass),
        }

    @staticmethod
    def _parse_decision(response: str) -> dict[str, Any]:
        try:
            decision = json.loads(response)
        except (TypeError, json.JSONDecodeError) as exc:
            raise GraphControllerError(
                "structured graph response is not valid JSON"
            ) from exc
        if not isinstance(decision, dict):
            raise GraphControllerError("structured graph response is not an object")
        return decision

    @staticmethod
    def _resolve_indices(
        snapshot: MergeCandidateSnapshot,
        indices: Any,
    ) -> tuple[list[int], dict[int, MergeCandidate]]:
        if not isinstance(indices, list):
            raise GraphControllerError("candidate_indices is not an array")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise GraphControllerError("candidate indices must be integers")
        by_index = {candidate.index: candidate for candidate in snapshot.candidates}
        if any(index not in by_index for index in indices):
            raise GraphControllerError("candidate index is not in the frozen snapshot")
        return indices, by_index

    def resolve_action(
        self,
        graph: ContextGraph,
        snapshot: MergeCandidateSnapshot,
        response: str,
        *,
        allow_pass: bool,
    ) -> dict[str, Any]:
        """Resolve a constrained decision to the existing graph handler API."""
        if self.graph_hash(graph) != snapshot.graph_hash:
            raise GraphControllerError("graph changed after candidate snapshot")
        decision = self._parse_decision(response)
        action = decision.get("action")
        if action not in {"merge", "prune", "add_edge", "select", "pass"}:
            raise GraphControllerError("unknown graph action")
        indices, by_index = self._resolve_indices(
            snapshot, decision.get("candidate_indices")
        )

        if action == "pass":
            if indices:
                raise GraphControllerError("pass requires no candidate indices")
            if not allow_pass:
                raise GraphControllerError("pass is illegal while graph is not saturated")
            return {"function": "pass", "arguments": {}}

        if action == "merge":
            if not 2 <= len(indices) <= 6:
                raise GraphControllerError("merge requires 2 to 6 candidate indices")
            # Preserve first-choice order while owning uniqueness in Python.
            indices = list(dict.fromkeys(indices))
            if len(indices) < 2:
                raise GraphControllerError("merge requires at least two unique candidates")
            summary = decision.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                raise GraphControllerError("merge summary is empty")
            return {
                "function": "merge",
                "arguments": {
                    "node_ids": ",".join(by_index[index].node_id for index in indices),
                    "summary": summary.strip(),
                },
            }

        required_arity = 2 if action == "add_edge" else 1
        if len(indices) != required_arity:
            raise GraphControllerError(
                f"{action} requires exactly {required_arity} candidate "
                f"{'indices' if required_arity > 1 else 'index'}"
            )
        if action == "add_edge":
            if indices[0] == indices[1]:
                raise GraphControllerError("add_edge requires distinct candidates")
            relation = decision.get("relation")
            if relation not in {"causal", "semantic", "temporal"}:
                raise GraphControllerError("add_edge relation is invalid")
            return {
                "function": "add_edge",
                "arguments": {
                    "source": by_index[indices[0]].node_id,
                    "target": by_index[indices[1]].node_id,
                    "relation": relation,
                },
            }
        return {
            "function": action,
            "arguments": {"node_id": by_index[indices[0]].node_id},
        }

    def resolve_merge(
        self,
        graph: ContextGraph,
        snapshot: MergeCandidateSnapshot,
        response: str,
    ) -> dict[str, str]:
        if self.graph_hash(graph) != snapshot.graph_hash:
            raise GraphControllerError("graph changed after candidate snapshot")
        decision = self._parse_decision(response)
        indices = decision.get("candidate_indices")
        summary = decision.get("summary")
        if not isinstance(indices, list):
            raise GraphControllerError("candidate_indices is not an array")
        if not 2 <= len(indices) <= 6:
            raise GraphControllerError("merge requires 2 to 6 candidate indices")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise GraphControllerError("candidate indices must be integers")
        # vLLM may select llguidance for this schema, and llguidance 1.7.6
        # rejects the JSON Schema ``uniqueItems`` keyword. Own uniqueness in
        # the controller instead: preserve the model's first-choice order and
        # reject only if fewer than two semantic choices remain.
        indices = list(dict.fromkeys(indices))
        if len(indices) < 2:
            raise GraphControllerError("merge requires at least two unique candidates")
        by_index = {candidate.index: candidate for candidate in snapshot.candidates}
        if any(index not in by_index for index in indices):
            raise GraphControllerError("candidate index is not in the frozen snapshot")
        if not isinstance(summary, str) or not summary.strip():
            raise GraphControllerError("merge summary is empty")
        node_ids = [by_index[index].node_id for index in indices]
        return {
            "node_ids": ",".join(node_ids),
            "summary": summary.strip(),
        }
