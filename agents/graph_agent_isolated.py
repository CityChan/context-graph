"""ContextGraph Agent Loop (Isolated Subgraph Variant).

Hierarchical/spatially-isolated version of context_graph_agent. Each branch
runs on its own private ContextGraph (a subgraph), so the main agent's
trajectory only ever sees:

  * The main (parent) graph: query root, subtask nodes, branch summary nodes,
    and the main agent's own search/open_page observations.
  * **Not** the internal exploration of any branch.

When a branch returns, its child graph is collapsed into a SUMMARY node on
the parent and then released. This bounds the main agent's training sequence
length to O(main_turns) instead of O(main_turns x graph_size), eliminating
the OOM that the global-injection variant runs into on long-horizon tasks.

Trade-offs vs the global variant:
  + Main agent prompt size is bounded (≈ FoldAgent), no quadratic blow-up.
  + Each branch's subgraph is small and self-contained.
  + Cross-branch reasoning still possible at the parent level via add_edge
    between branch SUMMARY nodes.
  - Main agent cannot directly see branch-internal observations; it must
    rely on the summary the branch returned.
  - Branch agents currently do not invoke graph ops on their child graph
    (they only contribute observation nodes via the wrapped run_action).
    Adding branch-side graph ops is a follow-up.

This variant is registered as agent loop name ``context_graph_isolated_agent``.
"""

import os
import re
import time
import copy
import asyncio
from functools import partial
import random
from uuid import uuid4
from typing import Any, Union

from verl import DataProto
from .utils import Agent, select_env, truncate_text, is_weird, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics
from .prompts import create_chat, BRANCH_MESSAGE_SEARCH, BRANCH_MESSAGE, SUMMARY_PROMPT_CODE, SUMMARY_PROMPT_SEARCH
from .verifier import judge_scope
from .context_graph import ContextGraph, NodeType, NodeStatus, EdgeRelation


def print_chat(chat):
    chat_str = ""
    for turn in chat:
        if is_weird(str(turn)):
            chat_str += '# ' + turn['role'] + ' **CJK**\n\n' + turn['content'] + "\n\n---\n\n"
        else:
            chat_str += '# ' + turn['role'] + '\n\n' + turn['content'] + "\n\n---\n\n"
    return chat_str


def extract_fn_call(text):
    if text is None:
        return None
    func_matches = re.findall(r'<function=([^>]+)>', text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f'<function={last_function}>')
    text_after_last_func = text[last_func_pos:]
    params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', text_after_last_func, re.DOTALL))
    return {'function': last_function, 'arguments': params}


def extract_summary(text: str) -> str:
    matches = re.findall(r'<summary>(.*?)</summary>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def clean_response(response):
    if '<function=return>' in response:
        response = response.split('<function=return>')[-1]
    else:
        response = re.split(r'<\[[^\]]+\]>', response)[-1]
    return response


# ── Graph operation handlers (parent-graph only in isolated variant) ──

def handle_merge(graph: ContextGraph, fn_call: dict) -> str:
    node_ids_str = fn_call['arguments'].get('node_ids', '')
    summary = fn_call['arguments'].get('summary', '')
    node_ids = [nid.strip() for nid in node_ids_str.split(',') if nid.strip()]

    if len(node_ids) < 2:
        return f"[Error] merge requires at least 2 node IDs (got {len(node_ids)}).\n\n{graph.to_state_text()}"

    invalid = [nid for nid in node_ids if nid not in graph.nodes]
    if invalid:
        return f"[Error] Unknown node IDs: {invalid}.\n\n{graph.to_state_text()}"

    merged_id = graph.merge(node_ids, summary)
    if merged_id is None:
        return f"[Error] Could not merge nodes {node_ids}.\n\n{graph.to_state_text()}"

    return f"Merged {node_ids} into [{merged_id}].\n\n{graph.to_state_text()}"


def handle_add_edge(graph: ContextGraph, fn_call: dict) -> str:
    source = fn_call['arguments'].get('source', '').strip()
    target = fn_call['arguments'].get('target', '').strip()
    relation_str = fn_call['arguments'].get('relation', 'semantic').strip()

    relation_map = {
        'causal': EdgeRelation.CAUSAL,
        'semantic': EdgeRelation.SEMANTIC,
        'temporal': EdgeRelation.TEMPORAL,
    }
    relation = relation_map.get(relation_str, EdgeRelation.SEMANTIC)

    if source not in graph.nodes:
        return f"[Error] Unknown source node: {source}.\n\n{graph.to_state_text()}"
    if target not in graph.nodes:
        return f"[Error] Unknown target node: {target}.\n\n{graph.to_state_text()}"

    graph.add_edge(source, target, relation)
    return f"Added edge {source} --{relation_str}--> {target}.\n\n{graph.to_state_text()}"


def handle_select(graph: ContextGraph, fn_call: dict) -> str:
    node_id = fn_call['arguments'].get('node_id', '').strip()

    if node_id not in graph.nodes:
        return f"[Error] Unknown node: {node_id}.\n\n{graph.to_state_text()}"

    if not graph.select(node_id):
        return f"[Error] Cannot select {node_id} (pruned?).\n\n{graph.to_state_text()}"

    node = graph.nodes[node_id]
    content_preview = node.content[:500]
    return f"Focus shifted to [{node_id}] ({node.type.value}).\n\nContent:\n{content_preview}\n\n{graph.to_state_text()}"


def handle_prune(graph: ContextGraph, fn_call: dict) -> str:
    node_id = fn_call['arguments'].get('node_id', '').strip()

    if node_id not in graph.nodes:
        return f"[Error] Unknown node: {node_id}.\n\n{graph.to_state_text()}"

    if not graph.prune(node_id):
        return f"[Error] Cannot prune {node_id} (root node?).\n\n{graph.to_state_text()}"

    return f"Pruned [{node_id}].\n\n{graph.to_state_text()}"


GRAPH_OPS = {'merge', 'add_edge', 'select', 'prune'}


def make_graph_aware_run_action(env, child_graph: ContextGraph):
    """Wrap run_action so a branch's search/open_page results also become
    OBSERVATION nodes in its private subgraph.

    The branch agent's conversation history is unchanged; the wrapper only
    sniffs the response to figure out which tool was called and adds the
    matching node to the child graph. This is the only mutation point for
    branch-side graph state in the isolated variant.
    """
    async def wrapped(response):
        observation = await run_action(env, response)
        if observation is None:
            return None
        try:
            fn_call = extract_fn_call(response)
            if fn_call is not None:
                if fn_call['function'] == 'search':
                    child_graph.add_node(
                        observation[:500],
                        NodeType.OBSERVATION,
                        parent_id=child_graph.active_node_id,
                        edge_relation=EdgeRelation.TEMPORAL,
                        metadata={'tool': 'search', 'query': fn_call['arguments'].get('query', '')},
                    )
                elif fn_call['function'] == 'open_page':
                    child_graph.add_node(
                        observation[:800],
                        NodeType.OBSERVATION,
                        parent_id=child_graph.active_node_id,
                        edge_relation=EdgeRelation.CAUSAL,
                        metadata={'tool': 'open_page'},
                    )
        except Exception as e:
            print(f'[GRAPH ISOLATED] tracking observation in child graph failed: {e}')
        return observation
    return wrapped


async def process_item(
        item: DataProto,
        context: TaskContext,
) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
    """Isolated ContextGraph agent loop.

    Differences from graph_agent.process_item:
      1. Each ``branch`` call spawns a private child ContextGraph via
         ``parent_graph.spawn_child``. The branch agent's tool calls are
         wrapped to add observation nodes to that child graph only.
      2. When a branch returns, the child graph is collapsed into a SUMMARY
         node on the parent (and the child is released).
      3. Main agent's observations carry only the parent graph's state text
         (which stays small: query + subtask nodes + summary nodes), so
         per-turn injection is bounded.
    """
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer

    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)

    def _get(arr):
        import numpy as np
        v = np.asarray(arr)
        return v.item() if v.ndim == 0 else v[0]

    ability = _get(item.non_tensor_batch['ability'])
    uid = item.non_tensor_batch.get('uid', uuid4().hex)
    gen_uid = item.non_tensor_batch.get('gen_uid', None)

    EnvClass = select_env(ability, config)
    print(is_train, EnvClass, '[ISOLATED]')
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")

    workflow = _get(item.non_tensor_batch['extra_info']).get('workflow', None) or getattr(config.plugin, "workflow", "search_graph")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)

    branch_prompt = BRANCH_MESSAGE_SEARCH if 'search' in workflow else BRANCH_MESSAGE
    summary_prompt = SUMMARY_PROMPT_SEARCH if 'search' in workflow else SUMMARY_PROMPT_CODE

    max_turn = getattr(config.plugin, 'max_turn', 64) if config.plugin else 64
    max_session = getattr(config.plugin, "max_session", 5)
    if not is_train:
        max_session = getattr(config.plugin, "val_max_session", max_session)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    process_reward = getattr(config.plugin, "process_reward", None)
    if process_reward is not None and isinstance(process_reward, str) and process_reward.lower() == "none":
        process_reward = None
    max_traj = getattr(config.plugin, "max_traj", None)
    enable_summary = getattr(config.plugin, "enable_summary", False)

    lambda_compact = getattr(config.plugin, "lambda_compact", 0.1)
    lambda_cost = getattr(config.plugin, "lambda_cost", 0.005)

    llm_client = context.llm_client

    # ── Initialize parent ContextGraph ──
    graph = ContextGraph(tokenizer, namespace_prefix="n")
    query_text = env.instance_info['problem_statement']
    root_id = graph.add_node(query_text, NodeType.QUERY)
    branch_node_map = {}  # branch_name -> subtask_node_id
    branch_subgraph_stats = {}  # branch_name -> child graph stats (for logging)

    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    branches = []
    branch_tasks = {}
    branch_return = {}
    init_len = len(agent['main'].context())
    current = 'main'
    session_start_time = time.time()
    iteration = 0
    mask_rollout = True
    session_message = []

    while iteration < max_turn:
        if time.time() - session_start_time > session_timeout:
            print('[SESSION] Session Timeout')
            break

        iteration += 1

        if enable_summary and len(agent[current].context()) - init_len > config.response_length * 0.95:
            if len(agent) >= max_session:
                print('[SESSION] Session OOC after session', len(agent))
                break
            agent[current].rollback(k=2)
            agent[current].append({'role': 'assistant', 'content': ""})
            agent[current].append({'role': 'user', 'content': summary_prompt})
            session_message.append({'role': 'user', 'content': summary_prompt})
            response = await agent[current].step()
            session_message.append({'role': 'assistant', 'content': response})
            if response is None:
                break
            summary = extract_summary(response) or response
            graph.add_node(summary, NodeType.SUMMARY,
                          parent_id=graph.active_node_id,
                          edge_relation=EdgeRelation.TEMPORAL)
            next_session_prompt = (
                f"For this question, you have already made the following progress in previous session, "
                f"summarized as follow:\n\n{summary}\n\nNow continue work on it.")
            current = current + '+'
            agent[current] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
            agent[current].append({'role': 'assistant', 'content': ""})
            agent[current].append({'role': 'user', 'content': next_session_prompt})
            session_message.append({'role': 'user', 'content': next_session_prompt})

        response = await agent['main'].step()

        if response is None:
            break

        session_message.append({'role': 'assistant', 'content': response})
        fn_call = extract_fn_call(response)

        # ── Graph operations on parent graph ──
        if fn_call is not None and fn_call['function'] in GRAPH_OPS:
            handler = {
                'merge': handle_merge,
                'add_edge': handle_add_edge,
                'select': handle_select,
                'prune': handle_prune,
            }[fn_call['function']]
            observation = handler(graph, fn_call)
            # Track LLM-initiated ops separately for cost_penalty
            graph.explicit_op_count += 1
            print(f'[GRAPH ISOLATED] {fn_call["function"]} -> {observation[:100]}')

        # ── Branch: spawn isolated child subgraph ──
        elif fn_call is not None and fn_call['function'] == 'branch':
            if len(branches) + 1 > max_session:
                observation = f"You've already reached the limit of {len(branches)} branch calls. Continue working independently."
            else:
                description = fn_call['arguments'].get('description', 'Agent')
                message_to_branch = fn_call['arguments'].get('prompt', 'Empty prompt')
                print('[BRANCH ISOLATED]', description, len(agent['main'].context()))

                # 1. Create subtask node in PARENT graph
                subtask_id = graph.add_node(
                    f"Subtask: {description}",  # only the description, not the full prompt
                    NodeType.SUBTASK,
                    parent_id=graph.active_node_id,
                    edge_relation=EdgeRelation.DECOMPOSITION,
                )

                # 2. Spawn private child graph for this branch
                branch_idx = len(branches)
                child_prefix = f"b{branch_idx}_"
                child_graph = graph.spawn_child(subtask_id, prefix=child_prefix)
                # Child graph's root mirrors the subtask query
                child_root = child_graph.add_node(
                    f"Subtask: {description}\n{message_to_branch}",
                    NodeType.QUERY,
                )

                agent_name = f"#{branch_idx}-" + description.replace(' ', '_')
                branches.append(agent_name)
                branch_tasks[agent_name] = message_to_branch
                branch_node_map[agent_name] = subtask_id

                history = agent['main'].messages()
                agent[agent_name] = Agent(llm_client, history, tokenizer, config, prompt_turn=prompt_turn)
                branch_prompt_formatted = branch_prompt.format(message=message_to_branch)

                # Branch sees its OWN child graph state, not the parent
                graph_state_hint = f"\n\nYour subtask working graph:\n{child_graph.to_state_text()}"
                agent[agent_name].append({'role': 'user', 'content': branch_prompt_formatted + graph_state_hint})

                # 3. Branch runs with a graph-aware run_action that tracks
                #    its observations into the child graph (not the parent)
                wrapped_action = make_graph_aware_run_action(env, child_graph)
                agent_return = await agent[agent_name].react(
                    wrapped_action,
                    max_turn=max_turn,
                    max_tokens=getattr(config.plugin, "branch_len", None),
                    session_timeout=session_timeout - time.time() + session_start_time,
                    should_continue=lambda resp: '<function=return>' not in resp,
                    safe_finish=lambda
                        x: "You are in branch mode and cannot branch task or finish the task. Use the `return` tool to go back to the main agent." if '<function=finish>' in x or '<function=branch>' in x else None,
                    summary_prompt="The context limit has been exceeded for the branch. Please finish the sub task directly and clearly state the progress made and the pending jobs of the sub task. Only summarize the sub task progress, using the return tool.",
                    observation_prompt=f"* You are now in branch mode: {description}. Conduct the sub task based on instruction, and when you complete the assigned sub task, use return tool to return, do not perform action beyond the assigned sub task.",
                )
                iteration += agent_return['iteration']
                last_response = agent_return['last_response']
                session_message.extend(agent[agent_name].messages()[len(history):])

                fn_call_ret = extract_fn_call(last_response)
                branch_message = None
                if fn_call_ret is not None and fn_call_ret['function'] == 'return':
                    if 'message' in fn_call_ret['arguments']:
                        branch_message = fn_call_ret['arguments'].get('message', 'Empty message')
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                elif fn_call_ret is not None and fn_call_ret['function'] == 'finish':
                    if 'message' in fn_call_ret['arguments']:
                        branch_message = fn_call_ret['arguments'].get('message', 'Empty message')
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                if branch_message is None:
                    branch_message = f'Branch has finished its task. The last message was:\n\n{clean_response(last_response)}'

                # 4. Collapse child graph into a SUMMARY node on the parent.
                #    The summary content is the branch's return message; the
                #    subgraph's stats become metadata. The child graph itself
                #    is released so it does not get serialized into any future
                #    parent prompt.
                child_stats = graph.collapse_child(subtask_id) or {}
                branch_subgraph_stats[agent_name] = child_stats

                summary_id = graph.add_node(
                    branch_message[:2000],
                    NodeType.SUMMARY,
                    parent_id=subtask_id,
                    edge_relation=EdgeRelation.CAUSAL,
                    metadata={
                        'collapsed_from_branch': agent_name,
                        'child_graph_prefix': child_prefix,
                        **{f'child_{k}': v for k, v in child_stats.items()},
                    },
                )
                print(f'[BRANCH ISOLATED] Collapsed {agent_name}: child stats={child_stats}')

                # Only inject parent graph state if there are multiple branches
                # (so the agent can see what summaries exist for cross-branch reasoning).
                # Single-branch case degenerates to fold-like behavior (no graph noise).
                if len(branches) > 1:
                    observation = f'{branch_message}\n\n{graph.to_state_text()}'
                else:
                    observation = branch_message
                branch_return[agent_name] = branch_message

        # ── Regular tools on main agent: add observation to PARENT graph ──
        else:
            observation = await run_action(env, response)
            if observation is None:
                mask_rollout = False
                break

            if fn_call is not None:
                if fn_call['function'] == 'search':
                    graph.add_node(
                        observation[:500],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.TEMPORAL,
                        metadata={'tool': 'search', 'query': fn_call['arguments'].get('query', '')},
                    )
                elif fn_call['function'] == 'open_page':
                    graph.add_node(
                        observation[:800],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.CAUSAL,
                        metadata={'tool': 'open_page'},
                    )
                elif fn_call['function'] == 'action':
                    graph.add_node(
                        observation[:300],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.TEMPORAL,
                        metadata={'tool': 'action', 'command': fn_call['arguments'].get('command', '')[:100]},
                    )

        # ── Auto graph operations on PARENT graph (only) ──
        # Parent graph stays small (subtask + summary + main observations),
        # so the same heuristic thresholds work fine.
        auto_pruned = graph.auto_prune_low_value(max_active=12)
        if auto_pruned:
            print(f'[GRAPH ISOLATED AUTO] Pruned low-value nodes: {auto_pruned}')

        auto_edges = graph.auto_connect_semantic(keyword_overlap_threshold=3)
        if auto_edges:
            print(f'[GRAPH ISOLATED AUTO] Added semantic edges: {auto_edges}')

        if fn_call and fn_call.get('function') == 'branch':
            auto_merged = graph.auto_merge_similar(similarity_threshold=2)
            if auto_merged:
                print(f'[GRAPH ISOLATED AUTO] Merged: {auto_merged}')

        if agent['main'].chat[-1]['role'] == 'user':
            print('[ROLE ERROR]')
            print(agent['main'].chat[-1])
            agent['main'].append({'role': 'assistant', 'content': str(response)})

        if process_reward:
            observation = truncate_text(observation, max_lines=100, merge_repeat=True, merge_num=4)

        # Auto-compress: when graph exceeds threshold, auto-merge oldest observations
        # into a summary. Agent doesn't need to learn merge — it happens automatically.
        # This keeps prompt length bounded while preserving key info.
        if len(graph.active_nodes) > 6:
            obs_nodes = [
                nid for nid, n in graph.nodes.items()
                if n.type == NodeType.OBSERVATION and n.is_active()
            ]
            if len(obs_nodes) >= 3:
                to_merge = obs_nodes[:3]
                contents = [graph.nodes[nid].content[:100] for nid in to_merge]
                auto_summary = "Explored: " + " | ".join(contents)
                merged_id = graph.merge(to_merge, auto_summary)
                if merged_id:
                    print(f'[GRAPH AUTO-MERGE] {to_merge} → {merged_id} ({len(graph.active_nodes)} active)')

        agent['main'].append({'role': 'user', 'content': observation})
        session_message.append({'role': 'user', 'content': observation})

    env.stats['session_time'] = time.time() - session_start_time

    # ── Reward computation ──
    print('[TASK] Task Finish, Start Reward [ISOLATED]')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent['main'].messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    graph_rewards = graph.compute_graph_reward(
        score[1], lambda_compact=lambda_compact, lambda_cost=lambda_cost
    )
    print(f'[GRAPH REWARD ISOLATED] {graph_rewards} | branch_subgraphs={branch_subgraph_stats}')

    outs = []
    env.stats['get_final_score'] = score[1]
    env.stats['traj_num'] = len(agent)
    env.stats['main_len'] = min(len(agent['main'].context()) - init_len, config.response_length)
    env.stats['total_token'] = len(tokenizer.encode(print_chat(user_prompt + session_message)))
    env.stats['main_turn'] = len(agent['main'].messages())
    env.stats['is_branch'] = int(len(agent) > 1)
    env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
    env.stats['use_all_branch'] = int(len(branches) + 1 > max_session)
    env.stats['graph_n_nodes'] = len(graph.nodes)
    env.stats['graph_n_active'] = len(graph.active_nodes)
    env.stats['graph_n_edges'] = len(graph.edges)
    env.stats['graph_ops'] = graph.operation_count
    env.stats['graph_n_summaries'] = graph_rewards.get('n_summaries', 0)
    env.stats['graph_reward'] = graph_rewards.get('graph_reward', score[1])
    # Isolated-variant specific stats: aggregate child subgraph sizes
    env.stats['isolated_n_subgraphs'] = len(branch_subgraph_stats)
    env.stats['isolated_total_subgraph_nodes'] = sum(s.get('n_total', 0) for s in branch_subgraph_stats.values())
    env.stats['isolated_total_subgraph_obs'] = sum(s.get('n_observations', 0) for s in branch_subgraph_stats.values())

    if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
        mask_rollout = False
    if score[1] > 0:
        mask_rollout = False

    is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
    if getattr(config.plugin, "must_finish", None):
        if not is_finish:
            score = ('', 0)

    # ── Process rewards (parallel logic to graph_agent.py) ──
    if process_reward and is_train:
        mask_rollout = False
        env.stats['concise_main'] = int(len(agent['main'].context()) - init_len <= config.response_length * 0.5)

        if 'cjk' in process_reward:
            env.stats['is_cjk'] = 0
            for name in agent:
                for i, turn in enumerate(agent[name].chat):
                    if is_weird(str(turn)):
                        print('[CJK ERROR]')
                        env.stats['is_cjk'] = 1
                        agent[name].set_process_reward(i, -1)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0)

        if score[1] > 0:
            GRAPH_OP_MARKERS = ('<function=merge>', '<function=prune>', '<function=select>', '<function=add_edge>')
            if len(agent['main'].context()) - init_len > config.response_length * 0.5:
                bad_turn = [i for i, turn in enumerate(agent['main'].messages()) if
                            '<function=branch>' not in str(turn) and '<function=finish>' not in str(turn)
                            and not any(m in str(turn) for m in GRAPH_OP_MARKERS)]
                agent['main'].set_process_reward(bad_turn, -1)

            # In isolated variant, do NOT penalize successful trajectories that
            # skip graph ops. This lets the agent degenerate to FoldAgent behavior
            # on simple tasks (no graph ops needed → no penalty → same reward as fold).
            # Graph ops are encouraged via positive per-turn rewards (merge=+0.2 etc.)
            # rather than penalizing their absence.

            # Graph reward is now outcome-only (via compute_graph_reward).
            # Per-turn token-level rewards removed: they caused tiny group std
            # in GRPO → advantage explosion (±15000) → gradient explosion.
            # See: usage_bonus in context_graph.py compute_graph_reward().

            if 'scope' in process_reward:
                env.stats['scope_judge'] = 1
                for name in branches:
                    assigned_task = branch_tasks[name]
                    return_message = branch_return[name]
                    is_focus, justification = await judge_scope(assigned_task, return_message)
                    if is_focus < 0:
                        print(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 1 - 0.2)
                        env.stats['scope_judge'] = 0
                    elif is_focus > 0:
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], 0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 1 + 0.2)

            for name in branches:
                for i, turn in enumerate(agent[name].chat):
                    ERR_MARKERS = (
                        'Failed to validate tool call',
                        'Failed to parse tool call',
                        'You are in branch mode and cannot branch task or finish the task.',
                        'No function call was detected in the model response',
                        '[Error] The "search" function requires a "query" argument',
                        '[Error] The "open_page" function requires either a "docid" or a "url".',
                        '[Error] The function',
                    )
                    if any(m in str(turn) for m in ERR_MARKERS):
                        agent[name].set_process_reward(i - 1, -1)
        else:
            is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
            if 'drop_fail' in process_reward:
                if not is_finish:
                    for name in branches:
                        if 'cjk' in process_reward:
                            should_drop = True
                            for i, turn in enumerate(agent[name].chat):
                                if is_weird(str(turn)):
                                    agent[name].set_process_reward(i, -2)
                                    if 'flat' in process_reward:
                                        agent[name].set_cache('reward', -1)
                                    should_drop = False
                            if should_drop:
                                agent.pop(name)
                        else:
                            agent.pop(name)

            if 'reward_scope' in process_reward:
                env.stats['scope_judge'] = 1
                for name in branches:
                    if name not in agent:
                        continue
                    assigned_task = branch_tasks[name]
                    return_message = branch_return[name]
                    is_focus, justification = await judge_scope(assigned_task, return_message)
                    if is_focus < 0:
                        print(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0 - 0.2)
                        env.stats['scope_judge'] = 0
                    elif is_focus > 0:
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], 0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0 + 0.2)

    use_graph_reward = process_reward and 'graph' in process_reward

    for name in agent if is_train else ['main']:
        out = await agent[name].get_data()
        agent_reward = score[1]

        if use_graph_reward and name == 'main':
            agent_reward = graph_rewards.get('graph_reward', agent_reward)
        elif process_reward is not None and 'flat' in process_reward and 'reward' in agent[name].info_cache:
            agent_reward = agent[name].info_cache['reward']

        out = AgentLoopOutput(
            prompt_ids=out['prompt_ids'],
            response_ids=out['response_ids'],
            response_mask=out['response_mask'],
            response_logprobs=out['response_logprobs'],
            multi_modal_data={},
            reward_score=agent_reward,
            num_turns=out['num_turns'],
            metrics=AgentLoopMetrics(),
            extra_fields={
                'messages': out['messages'],
                'env_stats': copy.deepcopy(env.stats) if hasattr(env, 'stats') else {},
                'num_branches': len(branches),
                'branch_names': branches,
                'mask_rollout': mask_rollout,
                'is_finish': is_finish,
                'agent_name': name,
                'message_str': print_chat(session_message),
                'meta_info': f"N: {len(agent)} | {name} | G:{len(graph.nodes)}n/{len(graph.edges)}e [iso]",
                'process_reward_mask': out['process_reward_mask'],
                'uid': uid,
                'gen_uid': gen_uid,
                'graph_state': graph.to_state_text(),
                'graph_rewards': graph_rewards,
                'isolated_subgraph_stats': branch_subgraph_stats,
            }
        )
        outs.append(copy.deepcopy(out))

    if max_traj is not None and len(outs) > max_traj:
        idx = [0] + sorted(random.sample(range(1, len(outs)), k=max_traj - 1))
        outs = [outs[i] for i in idx]

    return outs
