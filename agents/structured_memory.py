"""Bounded, provenance-preserving structured memory for inference-time agents.

The memory is deliberately a sidecar to :mod:`agents.context_graph`.  Facts do
not become graph-operation candidates, so enabling structured memory cannot
increase the model-facing ``merge``/``prune`` action space.  Raw evidence stays
in ContextGraph nodes and archives; this module stores only normalized facts
and pointers back to that evidence.
"""

from __future__ import annotations

import json
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


def parse_json_object(text: str) -> Optional[dict[str, Any]]:
    """Parse one JSON object from a model response without accepting prose."""
    if not isinstance(text, str) or not text.strip():
        return None
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


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
                    },
                    "required": [
                        "subject", "predicate", "object", "importance",
                        "confidence", "quote",
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
            "missing_information": {"type": "array", "items": {"type": "string"}},
            "suggested_searches": {"type": "array", "items": {"type": "string"}},
            "can_answer": {"type": "boolean"},
            "confidence": {
                "type": "string", "enum": ["high", "medium", "low"],
            },
        },
        "required": [
            "missing_information", "suggested_searches", "can_answer", "confidence",
        ],
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
        self.missing_information: list[str] = []
        self.suggested_searches: list[str] = []
        self.can_answer = False
        self.confidence = "low"
        self._fact_counter = 0
        self._sequence = 0
        self._by_key: dict[tuple[str, str, str], str] = {}

    def _next_fact_id(self) -> str:
        self._fact_counter += 1
        return f"f{self._fact_counter}"

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
            existing_id = self._by_key.get(key)
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
                deduplicated += 1
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

        return {
            "added": added,
            "deduplicated": deduplicated,
            "links_added": links_added,
            "total_facts": len(self.facts),
        }

    def update_gaps(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        missing = payload.get("missing_information", [])
        searches = payload.get("suggested_searches", [])
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
        )[:1000]
        return (
            "Extract only task-relevant factual claims from this tool observation. "
            "Do not copy search-result speculation as fact. Keep quote text short and "
            "verbatim enough to locate the evidence. Use links only when a new fact "
            "clearly supports, contradicts, or is related to an existing fact ID.\n\n"
            f"TASK:\n{self.task}\n\nEVIDENCE NODE: {evidence_node_id}\n"
            f"SOURCE METADATA: {source}\n\nEXISTING FACTS:\n{existing}\n\n"
            f"OBSERVATION:\n{str(observation)[:12000]}"
        )

    def gap_prompt(self, max_facts: int = 32) -> str:
        facts = self.fact_catalog(max_facts=max_facts) or "(none)"
        return (
            "Analyze whether the structured evidence is sufficient to answer the task. "
            "List only concrete missing information and specific searches that could "
            "resolve it. Do not propose searches for information already present.\n\n"
            f"TASK:\n{self.task}\n\nFACTS:\n{facts}"
        )

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
        if not self.facts and not self.missing_information:
            return ""
        lines = ["[Structured fact memory; fact IDs point to raw evidence nodes]"]
        for fact in self.ranked_facts(query, max_facts=max_facts):
            provenance = ",".join(fact.evidence_node_ids[-2:])
            line = (
                f"- [{fact.id}] {fact.text()} "
                f"[{fact.importance}; conf={fact.confidence:.2f}; evidence={provenance}]"
            )
            lines.append(line)
        if self.missing_information:
            lines.append("[Information gaps]")
            lines.extend(f"- {gap}" for gap in self.missing_information[:5])
        if self.suggested_searches and not self.can_answer:
            lines.append("[Suggested searches]")
            lines.extend(f"- {query}" for query in self.suggested_searches[:3])
        return self._truncate("\n".join(lines), max_tokens)

    def stats(self) -> dict[str, Any]:
        return {
            "facts": len(self.facts),
            "links": len(self.links),
            "gaps": len(self.missing_information),
            "can_answer": int(self.can_answer),
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "contextgraph.structured_memory.v1",
            "task": self.task,
            "facts": [asdict(fact) for fact in self.facts.values()],
            "links": [asdict(link) for link in self.links],
            "missing_information": list(self.missing_information),
            "suggested_searches": list(self.suggested_searches),
            "can_answer": self.can_answer,
            "confidence": self.confidence,
        }


def structured_memory_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a structured-memory controller. Return only the JSON "
                "object required by the supplied response schema."
            ),
        },
        {"role": "user", "content": prompt},
    ]


__all__ = [
    "EvidenceFact",
    "FactLink",
    "StructuredFactMemory",
    "coerce_bool",
    "fact_extraction_schema",
    "gap_analysis_schema",
    "parse_json_object",
    "structured_memory_messages",
]
