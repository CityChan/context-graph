"""Bounded, provenance-preserving structured memory for inference-time agents.

The memory is deliberately a sidecar to :mod:`agents.context_graph`.  Facts do
not become graph-operation candidates, so enabling structured memory cannot
increase the model-facing ``merge``/``prune`` action space.  Raw evidence stays
in ContextGraph nodes and archives; this module stores only normalized facts
and pointers back to that evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


_IMPORTANCE_SCORE = {
    "critical": 3.0,
    "important": 2.0,
    "supplementary": 1.0,
}
_LINK_RELATIONS = {"supports", "contradicts", "related"}


def coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Parse bool-like Hydra and environment values without truthy-string bugs."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"invalid boolean setting: {value!r}")


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]+", str(text).lower())
        if len(token) > 1
    }


def _normalized(text: str) -> str:
    return " ".join(str(text).lower().split())


def _balanced_object_candidates(text: str) -> list[str]:
    """Return balanced object substrings, ignoring braces inside strings."""
    candidates: list[str] = []
    start: Optional[int] = None
    depth = 0
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if depth and char in {'"', "'"}:
            quote = char
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : index + 1])
                start = None
    return candidates


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    """Parse one JSON object from imperfect schema-constrained model output.

    vLLM usually enforces valid JSON, but some model/server combinations return
    fenced JSON, a Python-style dictionary, or reasoning followed by multiple
    objects.  Try each balanced object independently so one malformed prefix
    does not discard a later valid controller decision.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    fenced = re.findall(r"```(?:json|python)?\s*([\s\S]*?)\s*```", cleaned)
    search_spaces = fenced + [cleaned]
    for search_space in search_spaces:
        for candidate in _balanced_object_candidates(search_space):
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    value = ast.literal_eval(candidate)
                except (SyntaxError, ValueError):
                    continue
            if isinstance(value, dict):
                return value
    return None


class GapStepScheduler:
    """Deterministic approximately-periodic scheduler for epistemic checks."""

    def __init__(self, interval: int, jitter: int, seed_text: str):
        self.interval = max(0, int(interval))
        self.jitter = min(max(0, int(jitter)), max(0, self.interval - 1))
        seed = int.from_bytes(
            hashlib.sha256(str(seed_text).encode("utf-8")).digest()[:8], "big"
        )
        self._rng = random.Random(seed)
        self.next_turn: Optional[int] = None
        if self.interval:
            self.next_turn = self._next_after(0)

    def _next_after(self, turn: int) -> int:
        offset = self._rng.randint(-self.jitter, self.jitter) if self.jitter else 0
        return int(turn) + max(1, self.interval + offset)

    def due(self, turn: int) -> bool:
        return self.next_turn is not None and int(turn) >= self.next_turn

    def mark_attempt(self, turn: int) -> None:
        if self.interval:
            self.next_turn = self._next_after(turn)


def fact_extraction_schema(max_facts: int = 8) -> dict[str, Any]:
    """JSON schema for one batched observation-to-facts controller call."""
    max_facts = max(1, int(max_facts))
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "maxItems": max_facts,
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                        "importance": {
                            "type": "string",
                            "enum": ["critical", "important", "supplementary"],
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "quote": {"type": "string"},
                        "duplicate_of": {
                            "type": "string",
                            "description": (
                                "Existing fact ID expressing the same claim, or an "
                                "empty string when this is a new fact."
                            ),
                        },
                    },
                    "required": [
                        "subject", "predicate", "object", "importance",
                        "confidence", "quote", "duplicate_of",
                    ],
                    "additionalProperties": False,
                },
            },
            "links": {
                "type": "array",
                "maxItems": max_facts * 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "new_fact_index": {
                            "type": "integer", "minimum": 0, "maximum": max_facts - 1,
                        },
                        "existing_fact_id": {"type": "string"},
                        "relation": {
                            "type": "string",
                            "enum": sorted(_LINK_RELATIONS),
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "new_fact_index", "existing_fact_id", "relation", "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["facts", "links"],
        "additionalProperties": False,
    }


def gap_analysis_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "covered_aspects": {
                "type": "array", "maxItems": 8,
                "items": {"type": "string"},
            },
            "missing_information": {
                "type": "array", "maxItems": 5,
                "items": {"type": "string"},
            },
            "suggested_searches": {
                "type": "array", "maxItems": 3,
                "items": {"type": "string"},
            },
            "can_answer": {"type": "boolean"},
            "confidence": {
                "type": "string", "enum": ["high", "medium", "low"],
            },
            "reasoning": {"type": "string"},
        },
        "required": [
            "covered_aspects", "missing_information", "suggested_searches",
            "can_answer", "confidence", "reasoning",
        ],
        "additionalProperties": False,
    }


def plan_initialization_schema(max_goals: int = 8) -> dict[str, Any]:
    """Schema for the paper's initial plan and goal decomposition."""
    return {
        "type": "object",
        "properties": {
            "plan": {"type": "string"},
            "goals": {
                "type": "array",
                "maxItems": max(1, int(max_goals)),
                "items": {"type": "string"},
            },
        },
        "required": ["plan", "goals"],
        "additionalProperties": False,
    }


@dataclass
class EvidenceFact:
    id: str
    subject: str
    predicate: str
    object: str
    importance: str
    confidence: float
    quote: str
    evidence_node_ids: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    sequence: int = 0

    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


@dataclass(frozen=True)
class FactLink:
    source_fact_id: str
    target_fact_id: str
    relation: str
    reason: str = ""


@dataclass
class MemoryNode:
    id: str
    node_type: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryEdge:
    source_id: str
    target_id: str
    relation: str
    reason: str = ""


class StructuredFactMemory:
    """A bounded fact index that never mutates ContextGraph topology."""

    def __init__(
        self,
        task: str,
        *,
        tokenizer=None,
        max_facts: int = 128,
    ):
        self.task = str(task)
        self.tokenizer = tokenizer
        self.max_facts = max(1, int(max_facts))
        self.facts: dict[str, EvidenceFact] = {}
        self.links: list[FactLink] = []
        self.nodes: dict[str, MemoryNode] = {}
        self.structural_edges: list[MemoryEdge] = []
        self.plan_node_id: Optional[str] = None
        self.plan = ""
        self.goals: list[str] = []
        self.covered_aspects: list[str] = []
        self.missing_information: list[str] = []
        self.suggested_searches: list[str] = []
        self.can_answer = False
        self.confidence = "low"
        self.gap_reasoning = ""
        self._fact_counter = 0
        self._node_counter = 0
        self._sequence = 0
        self.revision = 0
        self._by_key: dict[tuple[str, str, str], str] = {}
        self._observation_nodes: dict[str, str] = {}

    @property
    def ready_to_answer(self) -> bool:
        """True only for an explicit positive gap decision with no open gaps."""
        return bool(self.can_answer and not self.missing_information)

    def _next_fact_id(self) -> str:
        self._fact_counter += 1
        return f"f{self._fact_counter}"

    def _next_node_id(self, prefix: str) -> str:
        self._node_counter += 1
        return f"m{prefix}{self._node_counter}"

    def plan_prompt(self) -> str:
        return (
            "Decompose the task into a concise high-level plan and independently "
            "verifiable information goals. Do not answer the task yet. Goals should "
            "be concrete enough to guide tool searches.\n\n"
            f"TASK:\n{self.task}"
        )

    def initialize_plan(self, payload: dict[str, Any]) -> None:
        """Initialize plan/goal nodes before exploration."""
        if self.plan_node_id is not None:
            return
        plan = " ".join(str((payload or {}).get("plan", "")).split())[:2000]
        if not plan:
            plan = f"Gather and verify the evidence required to answer: {self.task}"
        raw_goals = (payload or {}).get("goals", [])
        goals = (
            [" ".join(str(goal).split())[:500] for goal in raw_goals if str(goal).strip()]
            if isinstance(raw_goals, list) else []
        )[:8]
        self.plan = plan
        self.goals = goals
        plan_id = self._next_node_id("plan")
        self.plan_node_id = plan_id
        self.nodes[plan_id] = MemoryNode(
            id=plan_id, node_type="plan", content=plan,
            metadata={"is_root": True},
        )
        for goal in goals:
            goal_id = self._next_node_id("goal")
            self.nodes[goal_id] = MemoryNode(
                id=goal_id, node_type="goal", content=goal,
            )
            self.structural_edges.append(MemoryEdge(
                source_id=plan_id,
                target_id=goal_id,
                relation="contains",
                reason="plan contains goal",
            ))
        self.revision += 1

    def record_observation(
        self,
        *,
        evidence_node_id: str,
        tool_name: str,
        observation: str,
        source_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record action/observation structure while raw text stays in ContextGraph."""
        if not evidence_node_id or evidence_node_id in self._observation_nodes:
            return
        raw_metadata = dict(source_metadata or {})
        raw_metadata.pop("raw_content", None)
        metadata = json.loads(json.dumps(
            raw_metadata,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ))
        action_id = self._next_node_id("action")
        action_content = str(tool_name)
        if metadata:
            action_content += f": {json.dumps(metadata, ensure_ascii=False, sort_keys=True)[:500]}"
        self.nodes[action_id] = MemoryNode(
            id=action_id,
            node_type="action",
            content=action_content,
            metadata={"tool": str(tool_name), **metadata},
        )
        observation_id = self._next_node_id("obs")
        self.nodes[observation_id] = MemoryNode(
            id=observation_id,
            node_type="observation",
            content=str(observation)[:1000],
            metadata={
                "evidence_node_id": evidence_node_id,
                "tool": str(tool_name),
                "content_length": len(str(observation)),
            },
        )
        if self.plan_node_id:
            self.structural_edges.append(MemoryEdge(
                source_id=self.plan_node_id,
                target_id=action_id,
                relation="produces",
            ))
        self.structural_edges.append(MemoryEdge(
            source_id=action_id,
            target_id=observation_id,
            relation="produces",
            reason=f"result from {tool_name}",
        ))
        self._observation_nodes[evidence_node_id] = observation_id

    @staticmethod
    def _source_label(metadata: Optional[dict[str, Any]]) -> str:
        metadata = metadata or {}
        for key in ("url", "query", "command", "tool"):
            value = str(metadata.get(key, "") or "").strip()
            if value:
                return f"{key}={value[:240]}"
        return ""

    def add_extraction(
        self,
        payload: dict[str, Any],
        *,
        evidence_node_id: str,
        source_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Validate and ingest one batched extractor response.

        Exact normalized triples are merged deterministically.  Approximate
        duplicate decisions remain the extractor/controller's responsibility;
        invalid link targets are ignored rather than exposed as graph ops.
        """
        raw_facts = payload.get("facts", []) if isinstance(payload, dict) else []
        if not isinstance(raw_facts, list):
            raw_facts = []
        source = self._source_label(source_metadata)
        preexisting_fact_ids = set(self.facts)
        resolved_ids: list[Optional[str]] = []
        added = 0
        deduplicated = 0

        for raw in raw_facts:
            if not isinstance(raw, dict):
                resolved_ids.append(None)
                continue
            subject = " ".join(str(raw.get("subject", "")).split())[:300]
            predicate = " ".join(str(raw.get("predicate", "")).split())[:200]
            obj = " ".join(str(raw.get("object", "")).split())[:600]
            if not subject or not predicate or not obj:
                resolved_ids.append(None)
                continue
            key = (_normalized(subject), _normalized(predicate), _normalized(obj))
            duplicate_of = str(raw.get("duplicate_of", "") or "").strip()
            existing_id = self._by_key.get(key)
            if duplicate_of in preexisting_fact_ids:
                existing_id = duplicate_of
            if existing_id:
                fact = self.facts[existing_id]
                if evidence_node_id not in fact.evidence_node_ids:
                    fact.evidence_node_ids.append(evidence_node_id)
                if source and source not in fact.sources:
                    fact.sources.append(source)
                try:
                    new_confidence = min(
                        1.0, max(0.0, float(raw.get("confidence", 0)))
                    )
                    fact.confidence = max(fact.confidence, new_confidence)
                except (TypeError, ValueError):
                    pass
                new_importance = str(raw.get("importance", "supplementary")).lower()
                if _IMPORTANCE_SCORE.get(new_importance, 0) > _IMPORTANCE_SCORE.get(
                    fact.importance, 0
                ):
                    fact.importance = new_importance
                resolved_ids.append(existing_id)
                self._by_key[key] = existing_id
                deduplicated += 1
                observation_id = self._observation_nodes.get(evidence_node_id)
                if observation_id:
                    edge = MemoryEdge(
                        source_id=observation_id,
                        target_id=existing_id,
                        relation="contains",
                        reason="observation supports an existing fact",
                    )
                    if edge not in self.structural_edges:
                        self.structural_edges.append(edge)
                continue
            if len(self.facts) >= self.max_facts:
                resolved_ids.append(None)
                continue
            self._sequence += 1
            fact_id = self._next_fact_id()
            importance = str(raw.get("importance", "supplementary")).lower()
            if importance not in _IMPORTANCE_SCORE:
                importance = "supplementary"
            try:
                confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            quote = " ".join(str(raw.get("quote", "")).split())[:500]
            fact = EvidenceFact(
                id=fact_id,
                subject=subject,
                predicate=predicate,
                object=obj,
                importance=importance,
                confidence=confidence,
                quote=quote,
                evidence_node_ids=[evidence_node_id],
                sources=[source] if source else [],
                sequence=self._sequence,
            )
            self.facts[fact_id] = fact
            self._by_key[key] = fact_id
            resolved_ids.append(fact_id)
            added += 1
            observation_id = self._observation_nodes.get(evidence_node_id)
            if observation_id:
                self.structural_edges.append(MemoryEdge(
                    source_id=observation_id,
                    target_id=fact_id,
                    relation="contains",
                ))

        raw_links = payload.get("links", []) if isinstance(payload, dict) else []
        existing_links = {
            (link.source_fact_id, link.target_fact_id, link.relation)
            for link in self.links
        }
        links_added = 0
        if isinstance(raw_links, list):
            for raw in raw_links:
                if not isinstance(raw, dict):
                    continue
                index = raw.get("new_fact_index")
                target = str(raw.get("existing_fact_id", ""))
                relation = str(raw.get("relation", "")).lower()
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= len(resolved_ids)
                    or not resolved_ids[index]
                    or target not in preexisting_fact_ids
                    or target == resolved_ids[index]
                    or relation not in _LINK_RELATIONS
                ):
                    continue
                key = (resolved_ids[index], target, relation)
                if key in existing_links:
                    continue
                self.links.append(FactLink(
                    source_fact_id=resolved_ids[index],
                    target_fact_id=target,
                    relation=relation,
                    reason=" ".join(str(raw.get("reason", "")).split())[:300],
                ))
                existing_links.add(key)
                links_added += 1

        # A semantic merge is still a graph update: it adds provenance and may
        # raise confidence/importance even when it does not allocate a fact ID.
        changed = bool(added or deduplicated or links_added)
        if changed:
            self.revision += 1
        return {
            "added": added,
            "deduplicated": deduplicated,
            "links_added": links_added,
            "total_facts": len(self.facts),
            "changed": changed,
            "revision": self.revision,
        }

    def update_gaps(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        covered = payload.get("covered_aspects", [])
        missing = payload.get("missing_information", [])
        searches = payload.get("suggested_searches", [])
        self.covered_aspects = [
            " ".join(str(item).split())[:500]
            for item in covered if str(item).strip()
        ][:8] if isinstance(covered, list) else []
        self.missing_information = [
            " ".join(str(item).split())[:500]
            for item in missing if str(item).strip()
        ][:5] if isinstance(missing, list) else []
        self.suggested_searches = [
            " ".join(str(item).split())[:500]
            for item in searches if str(item).strip()
        ][:3] if isinstance(searches, list) else []
        self.can_answer = bool(payload.get("can_answer", False))
        confidence = str(payload.get("confidence", "low")).lower()
        self.confidence = confidence if confidence in {"high", "medium", "low"} else "low"
        self.gap_reasoning = " ".join(
            str(payload.get("reasoning", "")).split()
        )[:1000]

    def fallback_gap_analysis(self) -> dict[str, Any]:
        """Build conservative guidance when the learned analyzer is malformed.

        The fallback never declares the task answerable.  It only maps plan goals
        to lexical fact coverage, preserving forward progress without allowing a
        weak controller response to terminate the rollout prematurely.
        """
        fact_terms = [
            _terms(fact.text()) | _terms(fact.quote) for fact in self.facts.values()
        ]
        goals = self.goals or [self.task]
        covered: list[str] = []
        missing: list[str] = []
        for goal in goals:
            goal_terms = _terms(goal)
            overlaps = [len(goal_terms & terms) for terms in fact_terms]
            if goal_terms and overlaps and max(overlaps) >= min(2, len(goal_terms)):
                covered.append(goal)
            else:
                missing.append(f"Evidence needed for goal: {goal}")
        if not self.facts and not missing:
            missing = [f"Evidence needed to answer: {self.task}"]
        searches = [
            re.sub(r"^Evidence needed for goal:\s*", "", item)[:500]
            for item in missing[:3]
        ]
        return {
            "covered_aspects": covered[:8],
            "missing_information": missing[:5],
            "suggested_searches": searches,
            "can_answer": False,
            "confidence": "low",
            "reasoning": (
                "Deterministic fallback used because the learned gap analyzer "
                "did not return a valid structured decision."
            ),
        }

    def ranked_facts(self, query: str, max_facts: int = 12) -> list[EvidenceFact]:
        query_terms = _terms(query) | _terms(self.task)

        def score(fact: EvidenceFact) -> tuple[float, int]:
            fact_terms = _terms(fact.text()) | _terms(fact.quote)
            overlap = len(query_terms & fact_terms) / max(len(query_terms), 1)
            value = (
                4.0 * overlap
                + _IMPORTANCE_SCORE.get(fact.importance, 0.0)
                + fact.confidence
                + fact.sequence * 1e-6
            )
            return value, fact.sequence

        return sorted(self.facts.values(), key=score, reverse=True)[: max(1, int(max_facts))]

    def fact_catalog(self, max_facts: int = 24) -> str:
        lines = []
        for fact in self.ranked_facts(self.task, max_facts=max_facts):
            lines.append(
                f"[{fact.id}] {fact.text()} "
                f"(importance={fact.importance}, confidence={fact.confidence:.2f})"
            )
        return "\n".join(lines)

    def extraction_prompt(
        self,
        observation: str,
        *,
        evidence_node_id: str,
        source_metadata: Optional[dict[str, Any]] = None,
        max_existing: int = 12,
    ) -> str:
        existing = self.fact_catalog(max_existing) or "(none)"
        source = json.dumps(
            {
                key: value
                for key, value in (source_metadata or {}).items()
                if key != "raw_content"
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )[:1000]
        return (
            "Act as both a fact extractor and semantic relational integrator. "
            "Extract task-relevant candidate factual claims from this tool observation. "
            "Search-result snippets may be retained as candidates when clearly "
            "attributed; assign them lower confidence than directly opened primary "
            "sources. Never turn a query, unsupported inference, or model speculation "
            "into a fact. Keep quote text short and verbatim enough to locate the "
            "evidence. Compare every extracted fact "
            "against the existing fact catalog. Set duplicate_of to the matching fact "
            "ID when the claims are semantically equivalent even if wording differs; "
            "otherwise use an empty string. Emit links whenever a genuinely new fact "
            "supports, contradicts, or is meaningfully related to an existing fact. "
            "Contradictions must never be silently merged.\n\n"
            f"TASK:\n{self.task}\n\nEVIDENCE NODE: {evidence_node_id}\n"
            f"SOURCE METADATA: {source}\n\nEXISTING FACTS:\n{existing}\n\n"
            f"OBSERVATION:\n{str(observation)[:12000]}"
        )

    def gap_prompt(self, max_facts: int = 32) -> str:
        facts = self.fact_catalog(max_facts=max_facts) or "(none)"
        goals = "\n".join(f"- {goal}" for goal in self.goals) or "- Answer the task"
        relations = "\n".join(
            f"- [{link.source_fact_id}] {link.relation} [{link.target_fact_id}]"
            for link in self.links[:24]
        ) or "(none)"
        return (
            "Analyze whether the structured evidence is sufficient to answer the task. "
            "Treat covered_aspects as K_cov, missing_information as structural K_mis "
            "links between goals and evidence, and suggested_searches as Q_sug. List "
            "only concrete missing information and specific searches that resolve it. "
            "Do not propose searches for information already present. Set "
            "can_answer=true only when every goal is supported by cited fact IDs and "
            "missing_information is empty. Keep every list item and the reasoning "
            "concise; include only the minimum necessary items.\n\n"
            f"TASK:\n{self.task}\n\nGOALS:\n{goals}\n\nFACTS:\n{facts}\n\n"
            f"FACT RELATIONS:\n{relations}"
        )

    def _relation_lines(self, selected_ids: set[str], max_links: int = 12) -> list[str]:
        contradictions = [
            link for link in self.links
            if link.relation == "contradicts"
            and (
                link.source_fact_id in selected_ids
                or link.target_fact_id in selected_ids
            )
        ]
        other = [
            link for link in self.links
            if link.relation != "contradicts"
            and (link.source_fact_id in selected_ids or link.target_fact_id in selected_ids)
        ]
        lines = []
        for link in (contradictions + other)[:max_links]:
            reason = f": {link.reason}" if link.reason else ""
            lines.append(
                f"- [{link.source_fact_id}] {link.relation} "
                f"[{link.target_fact_id}]{reason}"
            )
        return lines

    def _truncate(self, text: str, max_tokens: int) -> str:
        max_tokens = max(1, int(max_tokens))
        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= max_tokens:
                return text
            try:
                return self.tokenizer.decode(token_ids[:max_tokens])
            except Exception:
                pass
        return " ".join(text.split()[:max_tokens])

    def render_context(
        self,
        query: str,
        *,
        max_facts: int = 12,
        max_tokens: int = 1024,
    ) -> str:
        if (
            not self.plan
            and not self.facts
            and not self.covered_aspects
            and not self.missing_information
            and not self.can_answer
        ):
            return ""
        lines = ["[Current STRUCTMEM state; this snapshot replaces older snapshots]"]
        if self.plan:
            lines.append(f"[Plan] {self.plan}")
        if self.goals:
            lines.append("[Goals]")
            lines.extend(f"- {goal}" for goal in self.goals[:8])
        ranked = self.ranked_facts(query, max_facts=max_facts)
        lines.append("[Verified facts; IDs point to raw ContextGraph evidence]")
        for fact in ranked:
            provenance = ",".join(fact.evidence_node_ids[-2:])
            line = (
                f"- [{fact.id}] {fact.text()} "
                f"[{fact.importance}; conf={fact.confidence:.2f}; evidence={provenance}]"
            )
            lines.append(line)
        relation_lines = self._relation_lines({fact.id for fact in ranked})
        if relation_lines:
            lines.append("[Fact relationships]")
            lines.extend(relation_lines)
        if self.covered_aspects:
            lines.append("[Covered aspects]")
            lines.extend(f"- {item}" for item in self.covered_aspects[:8])
        if self.missing_information:
            lines.append("[Information gaps]")
            lines.extend(f"- {gap}" for gap in self.missing_information[:5])
        if self.suggested_searches and not self.can_answer:
            lines.append("[Next-action guidance: search one unresolved item; avoid repeats]")
            lines.extend(f"- {query}" for query in self.suggested_searches[:3])
        if self.can_answer:
            lines.append(
                "[Readiness] The gap analyzer found sufficient evidence. Stop searching "
                "and submit the answer with the finish tool."
            )
        return self._truncate("\n".join(lines), max_tokens)

    def render_answer_context(
        self,
        *,
        max_facts: int = 24,
        max_tokens: int = 1024,
    ) -> str:
        """Render a compact graph-grounded view for final answer synthesis."""
        if not self.facts:
            return ""
        ranked = self.ranked_facts(self.task, max_facts=max_facts)
        buckets = {"critical": [], "important": [], "supplementary": []}
        for fact in ranked:
            provenance = ",".join(fact.evidence_node_ids[-3:])
            buckets.setdefault(fact.importance, []).append(
                f"- [{fact.id}] {fact.text()} "
                f"[conf={fact.confidence:.2f}; evidence={provenance}]"
            )
        lines = [
            "[STRUCTMEM evidence for final answer]",
            "Answer from these consolidated facts and relationships. Prefer critical "
            "facts, resolve contradictions explicitly, and do not invent missing facts.",
        ]
        for importance in ("critical", "important", "supplementary"):
            if buckets.get(importance):
                lines.append(f"[{importance.title()} facts]")
                lines.extend(buckets[importance])
        relation_lines = self._relation_lines({fact.id for fact in ranked}, max_links=20)
        if relation_lines:
            lines.append("[Fact relationships]")
            lines.extend(relation_lines)
        if self.missing_information:
            lines.append("[Unresolved gaps]")
            lines.extend(f"- {gap}" for gap in self.missing_information[:5])
        lines.append(f"[Gap confidence] {self.confidence}")
        return self._truncate("\n".join(lines), max_tokens)

    def stats(self) -> dict[str, Any]:
        return {
            "facts": len(self.facts),
            "links": len(self.links),
            "nodes": len(self.nodes) + len(self.facts),
            "structural_edges": len(self.structural_edges),
            "goals": len(self.goals),
            "actions": sum(node.node_type == "action" for node in self.nodes.values()),
            "observations": sum(
                node.node_type == "observation" for node in self.nodes.values()
            ),
            "gaps": len(self.missing_information),
            "can_answer": int(self.can_answer),
            "confidence": self.confidence,
            "revision": self.revision,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "contextgraph.structured_memory.v2",
            "task": self.task,
            "plan": self.plan,
            "goals": list(self.goals),
            "nodes": [asdict(node) for node in self.nodes.values()] + [
                {
                    "id": fact.id,
                    "node_type": "fact",
                    "content": fact.text(),
                    "metadata": {
                        "importance": fact.importance,
                        "confidence": fact.confidence,
                        "evidence_node_ids": list(fact.evidence_node_ids),
                    },
                }
                for fact in self.facts.values()
            ],
            "edges": [asdict(edge) for edge in self.structural_edges] + [
                {
                    "source_id": link.source_fact_id,
                    "target_id": link.target_fact_id,
                    "relation": link.relation,
                    "reason": link.reason,
                }
                for link in self.links
            ],
            "facts": [asdict(fact) for fact in self.facts.values()],
            "links": [asdict(link) for link in self.links],
            "covered_aspects": list(self.covered_aspects),
            "missing_information": list(self.missing_information),
            "suggested_searches": list(self.suggested_searches),
            "can_answer": self.can_answer,
            "confidence": self.confidence,
            "gap_reasoning": self.gap_reasoning,
            "revision": self.revision,
        }


def structured_memory_messages(
    prompt: str,
    schema: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    schema_instruction = ""
    if schema:
        schema_instruction = (
            " The required JSON Schema is: "
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
    return [
        {
            "role": "system",
            "content": (
                "You are a structured-memory controller. Return only the JSON "
                "object required by the supplied response schema. Do not emit "
                "analysis, markdown fences, or tool syntax. Prefer compact values "
                "and include only necessary array items so the JSON object always "
                "finishes within the generation budget."
                + schema_instruction
            ),
        },
        {"role": "user", "content": prompt},
    ]


__all__ = [
    "EvidenceFact",
    "FactLink",
    "GapStepScheduler",
    "MemoryEdge",
    "MemoryNode",
    "StructuredFactMemory",
    "coerce_bool",
    "fact_extraction_schema",
    "gap_analysis_schema",
    "plan_initialization_schema",
    "parse_json_object",
    "structured_memory_messages",
]
