"""Training-only document coverage credit and label-free branch history.

Coverage is a lexical document-retrieval proxy, not a claim that a fact is correct.
Gold IDs never enter graph state, branch feedback, or a model request.
"""
from __future__ import annotations

import math
import re

from .graph_rpo import GraphRPOEvaluatorError, STATE_CHANGING_GRAPH_OPS, _config_get, _scale_and_clip_delta


def gold_docids(extra):
    """Require an explicit normalized training field; never guess dataset columns."""
    values = extra.get("graph_rpo_gold_docids")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("evidence GraphRPO requires nonempty extra_info.graph_rpo_gold_docids")
    result = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError("graph_rpo_gold_docids must contain string/integer document IDs")
        result.add(str(value).strip())
    return result


def terms(text):
    return re.findall(r"\w+", str(text).casefold())


def similarity(left, right):
    a, b = set(terms(left)), set(terms(right))
    return len(a & b) / len(a | b) if a and b else 0.0


def source_shingles(text):
    text = re.sub(r"^\(This page was already seen in a previous search\..*?\)\s*", "", str(text), flags=re.S)
    words = terms(text)
    return {tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7))}


def expanded_documents(before, after):
    """Opening more of a previously seen page is progress, not a duplicate retry."""
    known = {}
    for doc in before:
        known.setdefault(str(doc['docid']), set()).update(source_shingles(doc['text']))
    return {str(doc['docid']) for doc in after if str(doc['docid']) in known
            and source_shingles(doc['text']) - known[str(doc['docid'])]}


def covered_documents(view, documents, gold):
    """Require 8 consecutive source words in the bounded view, not just a doc ID.

Only tool-returned text is eligible. Paraphrases may be missed; the metric is
deliberately conservative and does not treat archive membership as visibility.
"""
    words = terms(view)
    shingles = {tuple(words[i:i + 8]) for i in range(max(0, len(words) - 7))}
    covered = set()
    for doc in documents:
        identity = str(doc["docid"])
        if identity not in gold:
            continue
        if source_shingles(doc['text']) & shingles:
            covered.add(identity)
    return covered


def branch_history(entries, limit=6):
    """No oracle labels, no hard rejection, and no claim that no-result means false."""
    if not entries:
        return ""
    rows = ["[Previous branch attempts]",
            "These are attempts, not verified conclusions. Revisit only with new evidence or a refined query."]
    for entry in entries[-limit:]:
        rows.append(f"Task: {entry['task'][:300]} | new retrieved documents: {entry['new_documents']} | "
                    f"returned report (unverified): {entry['report'][:500]}")
    return "\n".join(rows)


def assign_evidence_credits(*, agent, graph_trace, gold, documents, plugin_config):
    """Credit the initiating branch/controller turn, even on failed episodes."""
    if not gold:
        raise ValueError("Evidence credit needs gold documents")
    threshold = float(_config_get(plugin_config, "graph_rpo_duplicate_threshold", 0.6))
    penalty = float(_config_get(plugin_config, "graph_rpo_duplicate_penalty", 0.2))
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("duplicate threshold must be in (0, 1]")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("duplicate penalty must be finite and nonnegative")
    metrics = {"graph_rpo_valid_edits": 0, "graph_rpo_creditable_edits": 0,
               "graph_rpo_credited_edits": 0, "graph_rpo_delta_sum": 0.0,
               "graph_rpo_delta_abs_sum": 0.0, "graph_rpo_evidence_gained": 0,
               "graph_rpo_evidence_lost": 0, "graph_rpo_branch_decisions": 0,
               "graph_rpo_duplicate_branches": 0}
    previous = []
    for event in graph_trace["events"]:
        op = event.get("op")
        if event.get("source") != "model" or not event.get("success"):
            continue
        if op not in STATE_CHANGING_GRAPH_OPS | {"branch", "pass"}:
            continue
        metrics["graph_rpo_valid_edits"] += 1
        turn = event.get("assistant_turn_index")
        if not isinstance(turn, int):
            raise GraphRPOEvaluatorError("Evidence decision lacks its initiating assistant turn")
        gained, lost, duplicate = set(), set(), False
        if op == "branch":
            seen = set(event["evidence_seen_before"])
            retrieved = set(event["evidence_retrieved"])
            novel = retrieved - seen
            gained = novel & gold
            task = str(event["args"].get("description", "")) + " " + str(event["args"].get("prompt", ""))
            overlap = max((similarity(task, old) for old in previous), default=0.0)
            duplicate = overlap >= threshold and not novel and not event.get('evidence_expanded_documents')
            previous.append(task)
            event.update(evidence_new_documents=sorted(novel), branch_similarity=overlap)
            metrics["graph_rpo_branch_decisions"] += 1
            metrics["graph_rpo_duplicate_branches"] += int(duplicate)
        elif op != "pass":
            # Freeze the same question-conditioned bounded memory probe on both sides.
            # Documents acquired later cannot retrospectively validate an earlier edit.
            count = event["evidence_document_count"]
            available = documents[:count]
            before = covered_documents(event["credit_view_before"], available, gold)
            after = covered_documents(event["credit_view_after"], available, gold)
            gained, lost = after - before, before - after
        raw = (len(gained) - len(lost)) / len(gold) - penalty * duplicate
        _, delta = _scale_and_clip_delta(raw,
            delta_scale=float(_config_get(plugin_config, "graph_rpo_delta_scale", 1.0)),
            delta_max=float(_config_get(plugin_config, "graph_rpo_delta_max", 1.0)))
        agent.add_graph_edit_credit(turn, delta)
        event.update(graph_rpo_credit_backend="evidence", graph_rpo_delta=delta,
                     graph_rpo_delta_unclipped=raw, graph_rpo_outcome_gated=False,
                     evidence_gold_count=len(gold), evidence_duplicate_penalty=penalty,
                     graph_rpo_delta_scale=float(_config_get(plugin_config, "graph_rpo_delta_scale", 1.0)),
                     graph_rpo_delta_max=float(_config_get(plugin_config, "graph_rpo_delta_max", 1.0)),
                     evidence_credit_version="document-coverage-v1",
                     evidence_gained=sorted(gained), evidence_lost=sorted(lost),
                     duplicate_without_new_documents=duplicate)
        metrics["graph_rpo_creditable_edits"] += 1
        metrics["graph_rpo_credited_edits"] += int(abs(delta) > 1e-12)
        metrics["graph_rpo_delta_sum"] += delta
        metrics["graph_rpo_delta_abs_sum"] += abs(delta)
        metrics["graph_rpo_evidence_gained"] += len(gained)
        metrics["graph_rpo_evidence_lost"] += len(lost)
    found = {str(d["docid"]) for d in documents} & gold
    metrics["graph_rpo_evidence_recall"] = len(found) / len(gold)
    metrics["graph_rpo_duplicate_branch_rate"] = metrics["graph_rpo_duplicate_branches"] / max(1, metrics["graph_rpo_branch_decisions"])
    return metrics
