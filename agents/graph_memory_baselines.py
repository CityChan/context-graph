"""Episode-local, source-guided MemoBrain and A-MEM evaluation adapters.

Independent implementations, not the authors' trained checkpoints or benchmark
stacks. See docs/graph_memory_baselines.md for pinned sources and adaptations.
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache

import numpy as np


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PROVENANCE = {
    "memobrain": {"source": "https://github.com/qhjqhj00/MemoBrain",
                  "revision": "82f16e17c28313a57bf95d83340142b96507f3d1"},
    "amem": {"source": "https://github.com/agiresearch/A-mem",
             "revision": "ceffb860f0712bbae97b184d440df62bc910ca8d",
             "embedding_model": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
             "index": "episode-local exact cosine; CPU; no shared Chroma collection"},
}

MEMORIZE = """Build a dependency graph from the observed research interaction.
Return JSON only: {"add_nodes":[{"tmp_id":"n1","kind":"evidence",
"thought":[{"role":"assistant","content":"search goal"},
{"role":"user","content":"findings with source docids"}]}],
"add_edges":[{"src":1,"dst":"n1","rationale":"supports task"}]}.
Create subtask or evidence nodes only. Edges express decomposition, refinement,
or evidence supporting a task. Refer to existing numeric IDs or new tmp_ids.
Use only observed evidence, preserve docids and uncertainty, avoid cycles.
Do not execute tools or answer the task. An empty patch is allowed."""
RECALL = """Maintain the reasoning graph: flush superseded/invalid nodes, fold
completed sub-trajectories into faithful notes. Return JSON only:
{"flush_ops":[{"id":2,"rationale":"superseded"}],
"fold_ops":[{"ids":[3,4],"rationale":"completed subtask","notes":[
{"role":"assistant","content":"summary of work"},
{"role":"user","content":"findings, source docids and unresolved issues"}]}]}.
Only modify active, unprotected nodes; never node 1 (original task). Each node
may occur in at most one operation. Preserve evidence dependencies. Empty lists
are valid if no change is justified. Do not invent facts or execute tools."""
ANALYZE = """Create an A-MEM note's semantic metadata for this observed research
episode. Return JSON only: {"keywords":["key concept"],"context":"one-sentence
description of topic and findings","tags":["category"]}. Preserve source IDs
and uncertainty. The original content is stored unchanged. No tool calls."""
EVOLVE = """Evolve an A-MEM note using its semantic nearest neighbors. Return JSON:
{"should_evolve":true,"actions":["strengthen","update_neighbor"],
"suggested_connections":["n1"],"tags_to_update":["tag"],
"new_context_neighborhood":["updated neighbor context"],
"new_tags_neighborhood":[["tag"]]}.
strengthen links the NEW note to existing neighbor IDs and updates its tags.
update_neighbor updates neighbors' context/tags; both arrays must match the
given neighbor order and length. Keep original values when no update is needed.
Use only observed facts, not speculation. Do not change original note content.
If no evolution is warranted, use should_evolve=false and empty arrays."""


def settings(plugin):
    defaults = {"memory_helper_max_tokens": 1024, "memobrain_recall_interval": 5,
                "memobrain_context_threshold": 16384, "amem_topk": 5,
                "amem_memory_tokens": 4096}
    result = {}
    for key, default in defaults.items():
        value = getattr(plugin, key, default)
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid {key}: {value}")
        result[key] = value
    return result


def json_object(text):
    text = text.rsplit("</think>", 1)[-1].strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def strings(value):
    if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
        raise ValueError("Expected a list of strings")
    return value


def notes(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Expected nonempty notes")
    for msg in value:
        if (not isinstance(msg, dict) or set(msg) != {"role", "content"}
                or msg["role"] not in {"assistant", "user"}
                or not isinstance(msg["content"], str) or not msg["content"].strip()):
            raise ValueError("Notes require assistant/user roles and nonempty content")
    return copy.deepcopy(value)


def validate_dag(nodes, edges):
    pending = set(nodes)
    while pending:
        roots = pending - {e["dst"] for e in edges if e["src"] in pending}
        if not roots:
            raise ValueError("Graph operation creates a cycle")
        pending -= roots


class MemoBrainMemory:
    def __init__(self, task):
        self.nodes = {1: dict(kind="task", thought=task, episodes=[], active=True)}
        self.edges, self.episodes = [], []
        self.events = []

    def append(self, pair):
        self.episodes.append(copy.deepcopy(pair))

    def protected(self):
        # Whole episodes, rather than upstream's individual-message boundaries.
        recent = {0, *range(max(0, len(self.episodes) - 2), len(self.episodes))}
        return {nid for nid, n in self.nodes.items()
                if nid == 1 or recent.intersection(n["episodes"])}

    def state(self):
        return dict(nodes=self.nodes, edges=self.edges, protected=sorted(self.protected()))

    def patch(self, patch):
        if set(patch) != {"add_nodes", "add_edges"}:
            raise ValueError("Expected add_nodes and add_edges")
        candidate = copy.deepcopy(self.nodes)
        edges, mapping = copy.deepcopy(self.edges), {}
        for node in patch["add_nodes"]:
            tmp = node["tmp_id"]
            if not isinstance(tmp, str) or not tmp or tmp.isdecimal() or tmp in mapping:
                raise ValueError("tmp_id must be a unique nonnumeric string")
            if node["kind"] not in {"subtask", "evidence"}:
                raise ValueError("Only subtask/evidence nodes may be added")
            nid = max(candidate) + 1
            mapping[tmp] = nid
            candidate[nid] = dict(kind=node["kind"], thought=notes(node["thought"]),
                                  episodes=[len(self.episodes) - 1], active=True)
        for edge in patch["add_edges"]:
            src, dst = (mapping.get(str(edge[k]), edge[k]) for k in ("src", "dst"))
            src = int(src) if isinstance(src, str) and src.isdecimal() else src
            dst = int(dst) if isinstance(dst, str) and dst.isdecimal() else dst
            if type(src) is not int or type(dst) is not int or src not in candidate or dst not in candidate:
                raise ValueError("Edge refers to an unknown node")
            if not candidate[src]["active"] or not candidate[dst]["active"]:
                raise ValueError("Edge refers to an inactive node")
            edges.append(dict(src=src, dst=dst, rationale=str(edge.get("rationale", ""))))
        # Validate the entire directed graph before publishing any mutation.
        validate_dag(candidate, edges)
        self.nodes, self.edges = candidate, edges

    def maintain(self, decision):
        if set(decision) != {"flush_ops", "fold_ops"}:
            raise ValueError("Expected flush_ops and fold_ops")
        candidate, edges = copy.deepcopy(self.nodes), copy.deepcopy(self.edges)
        used, protected = set(), self.protected()
        for op, ids in [(op, [op["id"]]) for op in decision["flush_ops"]] + [
                (op, op["ids"]) for op in decision["fold_ops"]]:
            if (not isinstance(ids, list) or not ids or any(type(n) is not int for n in ids)
                    or len(set(ids)) != len(ids) or set(ids) & (used | protected)
                    or any(n not in candidate or not candidate[n]["active"] for n in ids)):
                raise ValueError("Invalid, protected, inactive or overlapping maintenance target")
            used.update(ids)
            if "ids" in op:
                nid = max(candidate) + 1
                candidate[nid] = dict(kind="summary", thought=notes(op["notes"]),
                    episodes=sorted({e for n in ids for e in candidate[n]["episodes"]}), active=True)
                rewritten = []
                for edge in edges:
                    src = nid if edge["src"] in ids else edge["src"]
                    dst = nid if edge["dst"] in ids else edge["dst"]
                    if src != dst:
                        rewritten.append(dict(src=src, dst=dst, rationale=edge["rationale"]))
                edges = rewritten
            for n in ids:
                candidate[n]["active"] = False
                candidate[n]["operation"] = "fold" if "ids" in op else "flush"
        validate_dag(candidate, edges)
        self.nodes, self.edges = candidate, edges

    def history(self):
        result = []
        protected = {0, *range(max(0, len(self.episodes) - 2), len(self.episodes))}
        for idx, pair in enumerate(self.episodes):
            related = [n for n in self.nodes.values() if idx in n["episodes"]]
            # Failed/empty patches never lose raw evidence. Flush uses node notes,
            # as upstream does, while fold replaces the covered episodes once.
            if idx in protected or not related or any(n["active"] and n["kind"] != "summary" for n in related):
                result.extend(pair)
                continue
            summaries = [n for n in related if n["kind"] == "summary" and n["active"]]
            if summaries:
                for n in summaries:
                    if idx == max(n["episodes"]):
                        result.extend(n["thought"])
            else:
                for n in related:
                    if n.get("operation") == "flush":
                        result.extend(n["thought"])
        return copy.deepcopy(result)


@lru_cache(maxsize=1)
def embedding_model():
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    path = snapshot_download(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, local_files_only=True)
    return SentenceTransformer(path, device="cpu", local_files_only=True)


def embed(text):
    return np.asarray(embedding_model().encode(text, normalize_embeddings=True,
                                             show_progress_bar=False), dtype=float)


class AMemMemory:
    def __init__(self, encoder=embed):
        self.nodes, self.vectors, self.events = {}, {}, []
        self.encoder = encoder
        self.embedding_calls = 0

    def vector(self, content):
        value = np.asarray(self.encoder(content), dtype=float)
        if value.ndim != 1 or not np.isfinite(value).all() or not np.linalg.norm(value):
            raise RuntimeError("Invalid A-MEM embedding")
        self.embedding_calls += 1
        return value / np.linalg.norm(value)

    def nearest(self, query, k):
        if not self.nodes:
            return []
        query = self.vector(query)
        return sorted(self.nodes, key=lambda n: (-float(self.vectors[n] @ query), n))[:k]

    def add(self, content, analysis):
        if set(analysis) != {"keywords", "context", "tags"} or not isinstance(analysis["context"], str):
            raise ValueError("Expected keywords, context, tags")
        strings(analysis["keywords"])
        strings(analysis["tags"])
        nid = f"n{len(self.nodes) + 1}"
        vector = self.vector(content)
        self.nodes[nid] = dict(id=nid, content=content, **copy.deepcopy(analysis),
                               links=[], evolution_history=[])
        self.vectors[nid] = vector
        return nid

    def evolve(self, nid, neighbors, decision):
        required = {"should_evolve", "actions", "suggested_connections", "tags_to_update",
                    "new_context_neighborhood", "new_tags_neighborhood"}
        if set(decision) != required or type(decision["should_evolve"]) is not bool:
            raise ValueError("Invalid evolution schema")
        actions = strings(decision["actions"])
        if set(actions) - {"strengthen", "update_neighbor"}:
            raise ValueError("Unknown evolution action")
        if not decision["should_evolve"]:
            return
        candidate = copy.deepcopy(self.nodes)
        if "strengthen" in actions:
            links = strings(decision["suggested_connections"])
            if set(links) - set(neighbors):
                raise ValueError("Links must use the provided neighbor IDs")
            candidate[nid]["links"] = list(dict.fromkeys(links))
            candidate[nid]["tags"] = strings(decision["tags_to_update"])
        if "update_neighbor" in actions:
            contexts = strings(decision["new_context_neighborhood"])
            tags = decision["new_tags_neighborhood"]
            if not isinstance(tags, list) or len(contexts) != len(neighbors) or len(tags) != len(neighbors):
                raise ValueError("Evolution arrays must align with neighbor IDs")
            for other, description, labels in zip(neighbors, contexts, tags):
                node = candidate[other]
                node["evolution_history"].append(dict(context=node["context"], tags=node["tags"]))
                node["context"], node["tags"] = description, strings(labels)
        self.nodes = candidate

    def retrieve(self, query, k):
        roots = self.nearest(query, k)
        # Upstream raw retrieval expands each nearest note's links. Deduplicate.
        ids = list(dict.fromkeys(roots + [n for root in roots for n in self.nodes[root]["links"][:k]]))
        return [self.nodes[n] for n in ids]

    def state(self):
        return dict(nodes=self.nodes, embedding_calls=self.embedding_calls)
