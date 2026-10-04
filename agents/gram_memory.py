"""GRAM's document-stream state and graph operations (no benchmark labels here)."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
import re
import string
import unicodedata
import xml.etree.ElementTree as ET


def entity_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def parse_action(text: str, *, external_search: bool = False) -> tuple[str, str]:
    """Exactly one action; optional preceding think, no stray text/nested tags."""
    try:
        root = ET.fromstring("<root>" + text.strip() + "</root>")
    except ET.ParseError as exc:
        raise ValueError("Expected one well-formed XML action") from exc
    children = list(root)
    if root.text and root.text.strip():
        raise ValueError("Text outside action tags")
    if any(c.attrib or list(c) or (c.tail and c.tail.strip()) for c in children):
        raise ValueError("Attributes, nested tags or trailing text are not actions")
    if children and children[0].tag == "think":
        children = children[1:]
    allowed = {"memory_insert", "memory_update", "memory_search", "answer"}
    if external_search:
        allowed.update({"search", "open_page"})
    if len(children) != 1 or children[0].tag not in allowed:
        raise ValueError("Exactly one Insert, Update, Search or Answer is required")
    content = (children[0].text or "").strip()
    if not content:
        raise ValueError("Empty action")
    return children[0].tag, content


def token_f1(prediction: str, answers: list[str]) -> float:
    def tokens(value):
        value = unicodedata.normalize("NFKC", value).lower()
        value = "".join(c for c in value if c not in string.punctuation)
        return re.sub(r"\b(a|an|the)\b", " ", value).split()
    p = tokens(prediction)
    scores = []
    for answer in answers:
        g = tokens(answer)
        overlap = sum((Counter(p) & Counter(g)).values())
        scores.append(2 * overlap / (len(p) + len(g)) if p and g else float(p == g))
    return max(scores, default=0.0)


def triples(value) -> list[tuple[str, str, str]]:
    if not isinstance(value, list):
        raise ValueError("Triples must be a JSON list")
    out = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError("Each triple must have three strings")
        if any(not isinstance(s, str) or not s.strip() for s in row):
            raise ValueError("Triple components must be nonempty strings")
        out.append((row[0].strip(), entity_key(row[1]), row[2].strip()))
    return out


def cosine(a, b):
    if len(a) != len(b) or not a or not all(math.isfinite(x) for x in [*a, *b]):
        raise ValueError("Invalid entity embedding")
    norm = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    if not norm:
        raise ValueError("Zero entity embedding")
    return sum(x*y for x, y in zip(a, b)) / norm


class GraphMemory:
    def __init__(self):
        self.edges: dict[tuple[str, str, str], set[str]] = {}

    @property
    def entities(self):
        return sorted({x for edge in self.edges for x in (edge[0], edge[2])})

    def apply(self, add, remove, source: str, aliases=None):
        # Validate the entire transaction before touching graph state.
        aliases = aliases or {}
        names = {entity_key(name): name for name in self.entities}
        def canonical(name):
            name = aliases.get(name, name)
            return names.setdefault(entity_key(name), name)
        def normalize(rows):
            return [(canonical(s), r, canonical(o)) for s, r, o in triples(rows)]
        removals, additions = normalize(remove), normalize(add)
        if any(edge not in self.edges for edge in removals):
            raise ValueError("Update attempted to remove an absent triple")
        updated = {edge: set(sources) for edge, sources in self.edges.items()}
        for edge in removals:
            del updated[edge]
        for edge in additions:
            updated.setdefault(edge, set()).add(source)
        self.edges = updated

    def snapshot(self):
        return [{"triple": list(e), "sources": sorted(src)} for e, src in sorted(self.edges.items())]

    def observation(self):
        """Compact index; exact relational evidence is returned by Search."""
        incoming = Counter(e[2] for e in self.edges)
        outgoing = Counter(e[0] for e in self.edges)
        return {"entities": [{"name": name, "in_degree": incoming[name], "out_degree": outgoing[name]}
                             for name in self.entities],
                "relations": sorted({e[1] for e in self.edges}), "edge_count": len(self.edges)}

    def search(self, query: str, max_hops=2, top_k=12):
        """Deterministic lexical seeds followed by bounded directed paths."""
        if max_hops < 1 or top_k < 1:
            raise ValueError("Search limits must be positive")
        words = set(re.findall(r"\w+", entity_key(query)))
        ranked = sorted(self.edges, key=lambda e: (
            -len(words & set(re.findall(r"\w+", entity_key(" ".join(e))))), e))
        seeds = [e for e in ranked if words & set(re.findall(r"\w+", entity_key(" ".join(e))))]
        paths = [[edge] for edge in seeds[:top_k]]
        frontier = deque(paths)
        candidate_limit = top_k * (max_hops + 1)
        while frontier and len(paths) < candidate_limit:
            path = frontier.popleft()
            if len(path) >= max_hops:
                continue
            for edge in sorted(self.edges):
                visited = {path[0][0], *(e[2] for e in path)}
                if edge[0] == path[-1][2] and edge[2] not in visited:
                    extended = path + [edge]
                    paths.append(extended)
                    frontier.append(extended)
                    if len(paths) == candidate_limit:
                        break
        # Prefer relational chains when several seeds would fill the quota.
        paths.sort(key=lambda path: (-len(path), path))
        return [[list(edge) for edge in path] for path in paths[:top_k]]


@dataclass
class DocumentStream:
    question: str
    documents: list[dict]
    cursor: int = 0
    allow_empty: bool = False

    def __post_init__(self):
        if not isinstance(self.question, str) or not self.question.strip() or (not self.documents and not self.allow_empty):
            raise ValueError("A question and at least one document are required")
        ids = []
        for doc in self.documents:
            if set(doc) != {"id", "title", "text"} or any(not isinstance(x, str) for x in doc.values()):
                raise ValueError("Public documents contain only string id/title/text")
            ids.append(doc["id"])
        if len(set(ids)) != len(ids):
            raise ValueError("Document IDs must be unique")

    @property
    def current(self):
        return self.documents[self.cursor] if self.cursor < len(self.documents) else None

    def advance(self):
        if self.current is None:
            raise ValueError("The document stream is exhausted")
        self.cursor += 1
