"""Explicit model graph tools and shared snapshot rendering for optional maintenance."""

import copy
import json
import re
import uuid

from .agent_text import extract_fn_call
from .context_graph import GraphOpResult, NodeStatus, NodeType, EdgeRelation
from .graph_operations import GRAPH_OPS, handle_merge, handle_prune, handle_select, handle_add_edge


EXPLICIT_GRAPH_TOOLS = GRAPH_OPS | {"graph_read", "recall", "graph_compact"}
GRAPH_DECISION_PROFILE = 'graph_affordances_v2'


def configure_evaluation_protocol(config, method, protocol="controller"):
    """Resolve evaluator settings before writing manifests or creating workers."""
    from .token_budget import configure_budget
    configure_budget(config)
    if protocol not in {'controller', 'explicit_tools'}:
        raise ValueError('Unknown graph action protocol')
    if protocol == 'explicit_tools' and method != 'contextgraph':
        raise ValueError('explicit_tools requires method=contextgraph')
    if method != 'contextgraph':
        return config
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.graph_action_protocol = protocol
    if protocol == 'explicit_tools':
        plugin.update({
            'structured_graph_controller': False, 'controller_owned_tool_formatting': False,
            'consolidation_interval': 0, 'initial_consolidation_turn': 0,
            'auto_prune_max_active': 0, 'auto_bind_branch_edges': False,
            'enable_summary': False, 'enable_retrieval_memory': False,
            'inject_graph_state_after_action': True, 'graph_controller_counts_as_turn': True,
            'action_output_profile': 'typed_action_json_v1',
            'graph_decision_profile': GRAPH_DECISION_PROFILE,
            'apply_chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True},
        })
    return config


def action_protocol(plugin):
    value = getattr(plugin, "graph_action_protocol", None)
    if value is None:
        return "controller" if getattr(plugin, "structured_graph_controller", False) else "explicit_tools"
    if value not in {"controller", "explicit_tools"}:
        raise ValueError("graph_action_protocol must be controller or explicit_tools")
    return value


EXPLICIT_GRAPH_PROMPT = """
Graph ownership: YOU decide when and how to manage memory. Graph tools are
ordinary JSON actions, available on every action turn, including in branches.
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
Working-context space and cumulative episode consumption are DIFFERENT quantities.
graph_compact reclaims window space, but never refunds recorded computation.
In working_context evaluation there is no cumulative token cap derived from the
window size; turn and time limits still apply. Legacy runs may cap cumulative use.
merge/prune alone do not remove old messages from your working conversation.
select changes focus for subsequent observations; add_edge records a relationship.
Neither is a compression operation.

Choose by the evidence and the cost, not a tool quota:
- Before repeating a search or measurement, inspect a relevant existing node with
  graph_read when its preview is insufficient. In a branch, this can recover
  the parent's evidence without repeating an environment action.
- When several nodes contain overlapping findings you have read, merge them into
  a concise note preserving source IDs, measured values, uncertainty and pending work.
  Never merge unseen previews as though they were verified facts.
- Prune a demonstrated dead end or obsolete state only when it is not needed for
  the current task. The raw node remains recoverable with graph_read/recall.
- If long/redundant history obstructs further work, use merge/prune to prepare a
  smaller active graph, then graph_compact if its snapshot fits the shown limits.
  If history is short, facts are still distinct, or little episode budget remains,
  continue a useful task action, return your branch findings, or finish instead.

Illustrative sequence, NOT a required workflow (IDs and sources below are fictional):
Suppose you have read n2: "source A measured 20 C" and n3: "source B confirms
20 C; pressure still unknown", and their raw pages dominate the working context.
One turn: {"tool":"merge","arguments":{"node_ids":["n2","n3"],"summary":"A and B agree on 20 C. Pressure remains unknown."}}
After receiving the new node ID and checking that the smaller snapshot fits,
a later turn: {"tool":"graph_compact","arguments":{}}
Then continue investigating pressure. Preserve the measured value and sources;
do not prune unrelated evidence or invent a pressure value.

Examples:
{"tool":"graph_read","arguments":{"offset":0}}
{"tool":"graph_read","arguments":{"node_id":"n3","offset":0}}
{"tool":"recall","arguments":{"node_id":"n3"}}
{"tool":"graph_compact","arguments":{}}
"""


def explicit_graph_prompt(plugin):
    """Keep legacy prompts stable; describe enabled full-profile maintenance truthfully."""
    if getattr(plugin, 'graph_rl_guard_profile', None) == 'prune_search_v1':
        return EXPLICIT_GRAPH_PROMPT + ('\nRepeated successful searches with the same normalized query and effective topk '
            'are blocked across MAIN and BRANCH, even after prune/compact. Failed searches may be retried. '
            'A blocked request still consumes a turn and creates no evidence node. Use graph_read/recall '
            'to recover stored evidence. Prune only evidence you judge unnecessary; do not prune to meet a quota.')
    if getattr(plugin, 'graph_rl_guard_profile', None) != 'graph_management_full_v1':
        return EXPLICIT_GRAPH_PROMPT
    prune = int(getattr(plugin, 'gm_auto_prune_max_active', 0))
    auto = bool(getattr(plugin, 'gm_auto_compact', False))
    chars = int(getattr(plugin, 'graph_compact_node_chars', 0))
    maintenance = ('There are no scheduled controller turns, automatic merges or semantic edges. '
                   f'Framework low-value auto-prune target: {prune if prune else "disabled"}. '
                   f'Framework automatic compact: {"enabled" if auto else "disabled"}. '
                   'Framework maintenance is separate from your actions.')
    text = EXPLICIT_GRAPH_PROMPT.replace(
        'There are no scheduled graph checkpoints, automatic prune/merge/semantic edges,\n'
        'automatic branch archival, or automatic context summaries.', maintenance)
    cap = (f'Each node content is capped at {chars} characters in the snapshot; use graph_read '
           'to recover the full stored content.' if chars else 'Node contents are not character-truncated.')
    text = text.replace('ALL currently active graph contents.', 'ALL currently active nodes. ' + cap)
    text = text.replace('Only this explicit call shortens conversation history.',
                        'Enabled framework automatic compact can also replace conversation history.')
    text = text.replace('but makes no\nsemantic decisions.',
                        'and applies only the configured automatic maintenance.')
    return text


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


def compaction_plan(agent, graph):
    """Read-only rendering shared by the model-visible estimate and executor."""
    active = sorted(graph.active_nodes, key=lambda n: (n.id != graph.active_node_id, n.id))
    char_limit = int(getattr(agent.config.plugin, 'graph_compact_node_chars', 0))
    if char_limit < 0:
        raise ValueError('graph_compact_node_chars must be nonnegative')
    def content(node):
        raw = node.metadata.get('raw_content', node.content)
        return raw[:char_limit] if char_limit else raw
    snapshot = json.dumps({
        "focus": graph.active_node_id,
        "nodes": [{"id": n.id, "type": n.type.value,
                   "content": content(n),
                   **({'content_truncated': len(content(n)) < len(n.metadata.get('raw_content', n.content)),
                       'raw_content_chars': len(n.metadata.get('raw_content', n.content))} if char_limit else {}),
                   "metadata": {k: v for k, v in n.metadata.items() if k != 'raw_content'}} for n in active],
        "edges": [{"source": e.source, "target": e.target, "relation": e.relation.value}
                  for e in graph.active_edges],
    }, ensure_ascii=False)
    # For branches this includes their assigned task, not the main's full history.
    prompt = copy.deepcopy(getattr(agent, "graph_base_prompt", agent.chat[:agent.prompt_turn]))
    label = 'Active graph' if getattr(agent.config.plugin, 'graph_rl_guard_profile', None) == 'graph_management_full_v1' else 'Model-selected active graph'
    prompt[-1]["content"] += f"\n\n[{label}]\n" + snapshot
    cost = len(agent._render_prefix(prompt))
    base_cost = len(agent._render_prefix(getattr(agent, "graph_base_prompt", agent.chat[:agent.prompt_turn])))
    charge = max(cost - base_cost, 0)
    return prompt, cost, charge


def compaction_prompt_limit(agent):
    """Window-mode snapshots are not limited by the initial prompt allocation."""
    from .token_budget import window_budget
    if not window_budget(agent.config):
        return agent.config.prompt_length
    window = getattr(agent, 'working_context_limit', None) or (agent.config.prompt_length + agent.config.response_length)
    # Match the executor's strict headroom guard, including the generation prefix.
    return max(0, window - len(agent.get_generation_prompt()) - 129)


def decision_state(graph, agent):
    """Expose resource facts, without choosing targets or scheduling graph work."""
    _, cost, charge = compaction_plan(agent, graph)
    remaining = agent.remaining_generation_tokens()
    from .token_budget import window_budget
    use_window = window_budget(agent.config)
    window = getattr(agent, 'working_context_limit', None) or (agent.config.prompt_length + agent.config.response_length)
    room = window - cost - len(agent.get_generation_prompt())
    limit = compaction_prompt_limit(agent)
    compact_fits = cost <= limit and (room > 128 if use_window else charge + 128 < remaining)
    budget_text = (f'window_tokens_remaining={remaining}; episode_token_limit=none; '
                   if use_window else f'episode_tokens_remaining={remaining}; ')
    return (graph.to_state_text() + '\n[Graph resource snapshot before this feedback is appended] '
            f'working_context_tokens={len(agent.context())}; {budget_text}'
            f'compact_prompt_tokens={cost}/{limit}; '
            f'compact_token_charge={charge}; '
            f'compact_fits_now={compact_fits}. '
            'Feedback and the next action also cost tokens. Only graph_compact replaces old history; '
            'merge/prune alone do not. Recorded computation is never refunded.')


def compact_context(agent, graph, *, source='model'):
    """Start an immutable segment and record whether model or framework requested it."""
    from .utils import AgentContext

    prompt, cost, charge = compaction_plan(agent, graph)
    from .exact_context import checksum
    fingerprint = checksum(prompt)
    if (getattr(agent.config.plugin, 'graph_reject_repeated_compact', False)
            and getattr(agent, 'last_graph_compact_fingerprint', None) == fingerprint):
        raise ValueError('repeated graph_compact with unchanged retained context')
    from .token_budget import window_budget
    use_window = window_budget(agent.config)
    window = getattr(agent, 'working_context_limit', None) or (agent.config.prompt_length + agent.config.response_length)
    if use_window and cost + len(agent.get_generation_prompt()) + 128 >= window:
        raise ValueError('compacted graph leaves insufficient working context')
    if not use_window and cost > agent.config.prompt_length:
        raise ValueError("active graph exceeds prompt budget; merge/prune more before graph_compact")
    if not use_window and charge + 128 >= agent.remaining_generation_tokens():
        raise ValueError("insufficient episode token budget for graph_compact")
    prefix_config = copy.deepcopy(agent.config)
    if use_window:
        # AgentContext otherwise silently truncates the snapshot to the original
        # prompt allocation. This changes only the segment, never the actor window.
        prefix_config.prompt_length = max(prefix_config.prompt_length, cost + 8)
    prefix = AgentContext(prompt, agent.tokenizer, prefix_config, prompt_turn=len(prompt),
                          chat_template_kwargs=agent.chat_template_kwargs)
    if prefix.messages() != prompt:
        raise ValueError('compacted graph prefix would be truncated')
    sessions = getattr(agent, "summary_sessions", [])
    sessions.append({"start": len(agent.chat), "prefix": prefix, "reason": source + "_graph_compact"})
    agent.summary_sessions = sessions
    agent.graph_context_tokens = getattr(agent, "graph_context_tokens", 0) + charge
    agent.context_uid = str(uuid.uuid4())
    agent.last_graph_compact_fingerprint = fingerprint
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
                        error=None if result.success else str(result), assistant_content=agent.chat[turn]['content'],
                        assistant_turn_index=turn, decision_context={"mode": "explicit_tool", "agent": agent_name})
    event["agent_name"] = agent_name
    maintenance = getattr(agent, 'sft_maintenance', None)
    if maintenance is not None and maintenance.active:
        event['decision_context']['sft_assistance'] = dict(maintenance.active)
    if attempted == 'graph_compact' and result.success:
        event['context_refresh'] = {'tokens_before': context_before, 'tokens_after': len(agent.context()),
                                    'session_index': len(agent.summary_sessions),
                                    'cumulative_refresh_tokens': agent.graph_context_tokens}
    if maintenance is not None:
        maintenance.executed(attempted, result.success, context_before, len(agent.context()))
    if penalize and getattr(agent.config.plugin, 'graph_rpo_credit_backend', None) == 'management':
        from .graph_management_reward import record_operation
        credit = record_operation(graph, agent, event, context_before, len(agent.context()))
        agent.add_graph_edit_credit(turn, credit)
    if not result.success and penalize:
        agent.set_process_reward(turn, -0.3)
        agent.additional_info[turn]['graph_invalid_penalty'] = -0.3
    runtime = getattr(agent, 'graph_management_runtime', None)
    if runtime is not None:
        runtime.edited(agent, event)
    prune_runtime = getattr(agent, 'prune_search_runtime', None)
    if prune_runtime is not None:
        prune_runtime.edited(agent, event)
    return result


def record_observation(graph, response, observation):
    call = extract_fn_call(response)
    if call is None:
        return None
    return graph.add_node(str(observation)[:500], NodeType.OBSERVATION,
                          parent_id=graph.active_node_id, edge_relation=EdgeRelation.TEMPORAL,
                          metadata={"tool": call["function"], "arguments": call['arguments'],
                                    "raw_content": str(observation)})
