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
