"""Opt-in ContextGraph wording and memory aids for long action-based environments.

Both are off by default, so existing protocols are unchanged.

* ``graph_vocabulary="active"``: the graph's active node is historically called the
  "focus" in tool descriptions, graph renders ("[FOCUS]", "focus=[n3]"), controller help
  and select feedback. In ScienceWorld ``focus on <object>`` is an irreversible task
  commitment, so this wording collides with an environment verb. The neutral wording only
  changes text; graph semantics are identical.
* ``action_repeat_memory``: when an environment action and its result exactly match earlier
  action nodes in the graph (including pruned ones), the latest feedback states how often it
  already happened. The memory is ContextGraph's own node history; nothing is blocked.
* ``commit_graph_evidence``: when the environment asks for confirmation of an irreversible commit
  (ScienceWorld ``scienceworld_commit_check``), ContextGraph attaches the observation lines from its
  nodes (including pruned ones) that mention the commit target.
"""

_MODE = {"vocabulary": "focus"}


def set_vocabulary(mode):
    if mode not in ("focus", "active"):
        raise ValueError(f"Unknown graph vocabulary: {mode}")
    _MODE["vocabulary"] = mode


def neutral():
    return _MODE["vocabulary"] == "active"


def normalize_command(command):
    return " ".join(str(command).lower().split())


def repeat_note(graph, node_id):
    """Feedback prefix when ``node_id`` repeats an earlier (command, result) pair, else ''."""
    node = graph.nodes.get(node_id)
    if node is None or node.metadata.get("tool") != "action":
        return ""
    command = normalize_command(node.metadata.get("command", ""))
    result = node.metadata.get("raw_content", node.content)
    earlier = [
        other.id for other in graph.nodes.values()
        if other.id != node_id and other.metadata.get("tool") == "action"
        and normalize_command(other.metadata.get("command", "")) == command
        and other.metadata.get("raw_content", other.content) == result
    ]
    if not earlier:
        return ""
    shown = ", ".join(earlier[-5:])
    return (
        f"[ContextGraph memory] The action `{command}` already produced this exact result "
        f"{len(earlier)} time(s) earlier in this episode (nodes {shown}). Repeating it will "
        "not change the simulator state; choose a different action that makes progress."
    )


def commit_evidence(graph, target, limit=6):
    """Observation lines stored in the graph that mention ``target`` (most recent last)."""
    target = normalize_command(target)
    if not target:
        return ""
    words = [w for w in target.split() if w.isalpha() and len(w) >= 4]
    keys = [target] + ([max(words, key=len)] if words and max(words, key=len) != target else [])
    for key in keys:
        lines, seen = [], set()
        for node in graph.nodes.values():
            if node.metadata.get("tool") != "action":
                continue
            raw = str(node.metadata.get("raw_content", node.content))
            if raw.startswith("[Commit check]"):
                continue
            for line in raw.splitlines():
                text = " ".join(line.split())
                if key in text.lower() and text not in seen:
                    seen.add(text)
                    lines.append(f"{node.id} (`{node.metadata.get('command', '')}`): {text[:240]}")
        if lines:
            return f"[ContextGraph evidence mentioning `{key}`]\n" + "\n".join(lines[-limit:])
    return f"[ContextGraph evidence] No stored observation mentions `{target}`."
