"""Model-initiated graph tools, without scheduled or heuristic maintenance."""

import copy
import json
import re
import uuid

from .agent_text import extract_fn_call
from .context_graph import GraphOpResult, NodeStatus, NodeType, EdgeRelation
from .graph_operations import GRAPH_OPS, handle_merge, handle_prune, handle_select, handle_add_edge


EXPLICIT_GRAPH_TOOLS = GRAPH_OPS | {"graph_read", "recall", "graph_compact"}


def action_protocol(plugin):
    value = getattr(plugin, "graph_action_protocol", None)
    if value is None:
        return "controller" if getattr(plugin, "structured_graph_controller", False) else "explicit_tools"
    if value not in {"controller", "explicit_tools"}:
        raise ValueError("graph_action_protocol must be controller or explicit_tools")
    return value


EXPLICIT_GRAPH_PROMPT = """
Graph ownership: YOU decide when and how to manage memory. Graph tools are
ordinary XML tools, available on every action turn, including in branches.
There are no scheduled graph checkpoints, automatic prune/merge/semantic edges,
automatic branch archival, or automatic context summaries. Continue task tools
when no graph change is useful. Do not use pass. Emit exactly one complete tool
call per response. IDs may come from graph_read as well as the latest state.

merge(node_ids, summary): fold 2-6 distinct active nodes into YOUR summary.
prune(node_id): archive an active non-root node; its raw content remains stored.
add_edge(source, target, relation): add causal/semantic/temporal relationship.
select(node_id): change focus to another active node.
recall(node_id): restore a folded/pruned node to the active graph.
graph_read(node_id, offset): read up to 2000 characters of any stored node,
including archived nodes. Omit node_id to list 20 nodes, including inactive IDs;
offset is a zero-based character offset for content, or row offset for the list.
graph_compact(): replace YOUR working conversation with the original task and
ALL currently active graph contents. First merge/prune to choose what to retain.
Only this explicit call shortens conversation history. It refuses an oversized
snapshot rather than silently dropping nodes. Old policy tokens stay in training.
Branches start with the original task, their assignment and shared graph state;
use graph_read to inspect evidence. Their observations remain after return.
The framework records observations/temporal/delegation links, but makes no
semantic decisions. Graph tools consume the same turn/token budget as task tools.

Examples:
<function=graph_read><parameter=offset>0</parameter></function>
<function=graph_read><parameter=node_id>n3</parameter><parameter=offset>0</parameter></function>
<function=recall><parameter=node_id>n3</parameter></function>
<function=graph_compact></function>
"""


def graph_call(text):
    """Recognize even malformed/mixed graph calls so they cannot reach the env."""
    names = re.findall(r"<function=([^>]+)>", (text or "").rsplit("</think>", 1)[-1])
    for name in names:
        if name in EXPLICIT_GRAPH_TOOLS:
            return {"function": name, "arguments": {}}
    return None


def parse_graph_call(text):
    visible = (text or "").rsplit("</think>", 1)[-1]
    names = re.findall(r"<function=([^>]+)>", visible)
    if len(names) != 1 or visible.count("</function>") != 1:
        raise ValueError("emit exactly one complete graph tool call")
    name = names[0]
    start = visible.index(f"<function={name}>")
    body = visible[start + len(f"<function={name}>"):]
    if not body.endswith("</function>"):
        body = body.rstrip()
    if not body.endswith("</function>"):
        raise ValueError("graph call must end with </function>")
    body = body[:-len("</function>")]
    matches = list(re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.S))
    if re.sub(r"<parameter=([^>]+)>(.*?)</parameter>", "", body, flags=re.S).strip():
        raise ValueError("malformed graph parameters")
    arguments = {m[1]: m[2].strip() for m in matches}
    if len(arguments) != len(matches):
        raise ValueError("duplicate graph parameter")
    required, optional = {
        "merge": ({"node_ids", "summary"}, set()),
        "prune": ({"node_id"}, set()), "select": ({"node_id"}, set()),
        "recall": ({"node_id"}, set()),
        "add_edge": ({"source", "target", "relation"}, set()),
        "graph_read": (set(), {"node_id", "offset"}),
        "graph_compact": (set(), set()),
    }[name]
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise ValueError("missing or unknown graph parameters")
    if any(not arguments[key] for key in required):
        raise ValueError("empty required graph parameter")
    if name == "add_edge" and arguments["relation"] not in {"causal", "semantic", "temporal"}:
        raise ValueError("unknown edge relation")
    return {"function": name, "arguments": arguments}


def compact_context(agent, graph):
    """Start a new immutable training segment only at the model's request."""
    from .utils import AgentContext

    active = sorted(graph.active_nodes, key=lambda n: (n.id != graph.active_node_id, n.id))
    snapshot = json.dumps({
        "focus": graph.active_node_id,
        "nodes": [{"id": n.id, "type": n.type.value,
                   "content": n.metadata.get("raw_content", n.content),
                   "metadata": {k: v for k, v in n.metadata.items() if k != 'raw_content'}} for n in active],
        "edges": [{"source": e.source, "target": e.target, "relation": e.relation.value}
                  for e in graph.active_edges],
    }, ensure_ascii=False)
    # For branches this includes their assigned task, not the main's full history.
    prompt = copy.deepcopy(getattr(agent, "graph_base_prompt", agent.chat[:agent.prompt_turn]))
    prompt[-1]["content"] += "\n\n[Model-selected active graph]\n" + snapshot
    cost = len(agent._render_prefix(prompt))
    base_cost = len(agent._render_prefix(getattr(agent, "graph_base_prompt", agent.chat[:agent.prompt_turn])))
    charge = max(cost - base_cost, 0)
    if cost > agent.config.prompt_length:
        raise ValueError("active graph exceeds prompt budget; merge/prune more before graph_compact")
    if charge + 128 >= agent.remaining_generation_tokens():
        raise ValueError("insufficient episode token budget for graph_compact")
    prefix = AgentContext(prompt, agent.tokenizer, agent.config, prompt_turn=len(prompt),
                          chat_template_kwargs=agent.chat_template_kwargs)
    sessions = getattr(agent, "summary_sessions", [])
    sessions.append({"start": len(agent.chat), "prefix": prefix, "reason": "model_graph_compact"})
    agent.summary_sessions = sessions
    agent.graph_context_tokens = getattr(agent, "graph_context_tokens", 0) + charge
    agent.context_uid = str(uuid.uuid4())
    return GraphOpResult("Working context rebuilt from every active node selected by your graph edits. Continue the task.", True)


def execute_graph_tool(graph, trace, agent, response, *, agent_name, turn_id, penalize):
    before = trace.capture(graph)
    context_before = len(agent.context())
    attempted = graph_call(response)["function"]
    call = {"function": attempted, "arguments": {}}
    turn = len(agent.chat) - 1
    agent.add_graph_edit_credit(turn, 0.0)
    try:
        call = parse_graph_call(response)
        name, args = call["function"], call["arguments"]
        if name in GRAPH_OPS:
            result = {"merge": handle_merge, "prune": handle_prune, "select": handle_select,
                      "add_edge": handle_add_edge}[name](graph, call)
        elif name == "recall":
            node = graph.nodes.get(args["node_id"])
            if node is None or node.is_active():
                raise ValueError("recall requires a stored inactive node")
            node.status = NodeStatus.ACTIVE
            graph.operation_count += 1
            result = GraphOpResult(f"Restored [{node.id}].\n{graph.to_state_text()}", True)
        elif name == "graph_read":
            offset = int(args.get("offset", "0"))
            if offset < 0:
                raise ValueError("offset must be nonnegative")
            if "node_id" in args:
                node = graph.nodes.get(args["node_id"])
                if node is None:
                    raise ValueError("unknown node ID")
                content = node.metadata.get("raw_content", node.content)
                result = GraphOpResult(f"[{node.id}] {node.status.value} chars={len(content)} offset={offset}\n"
                                      + f"Tool: {node.metadata.get('tool', '')} {node.metadata.get('arguments', {})}\n"
                                      + content[offset:offset + 2000], True)
            else:
                nodes = list(graph.nodes.values())
                result = GraphOpResult(f"Stored nodes {offset}:{min(offset + 20, len(nodes))}/{len(nodes)}\n"
                    + "\n".join(f"{n.id} {n.status.value} {n.type.value}: {n.content[:80]}" for n in nodes[offset:offset + 20]), True)
        else:
            result = compact_context(agent, graph)
    except (ValueError, KeyError) as exc:
        result = GraphOpResult(f"[Graph tool rejected] {exc}", False)
    graph.record_graph_op(result.success)
    event = trace.record(graph, before, turn_id=turn_id, source="model", op=attempted,
                        args=call["arguments"], success=result.success,
                        error=None if result.success else str(result), assistant_content=response,
                        assistant_turn_index=turn, decision_context={"mode": "explicit_tool", "agent": agent_name})
    event["agent_name"] = agent_name
    if attempted == 'graph_compact' and result.success:
        event['context_refresh'] = {'tokens_before': context_before, 'tokens_after': len(agent.context()),
                                    'session_index': len(agent.summary_sessions),
                                    'cumulative_refresh_tokens': agent.graph_context_tokens}
    if not result.success and penalize:
        agent.set_process_reward(turn, -0.3)
    return result


def record_observation(graph, response, observation):
    call = extract_fn_call(response)
    if call is None:
        return None
    return graph.add_node(str(observation)[:500], NodeType.OBSERVATION,
                          parent_id=graph.active_node_id, edge_relation=EdgeRelation.TEMPORAL,
                          metadata={"tool": call["function"], "arguments": call['arguments'],
                                    "raw_content": str(observation)})
