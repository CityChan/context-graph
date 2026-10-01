"""ContextGraph Agent Loop (Isolated Subgraph Variant).

Hierarchical/spatially-isolated version of context_graph_agent. Each branch
runs on its own private ContextGraph (a subgraph), so the main agent's
trajectory only ever sees:

  * The main (parent) graph: query root, subtask nodes, branch summary nodes,
    and the main agent's own tool observations.
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
import time
import copy
import asyncio
from functools import partial
import random
from uuid import uuid4
from typing import Any, Union

# ContextGraph isolated agent for code-execution tasks (ScienceAgentBench).
#
# Fork of graph_agent_isolated.py. Same graph state machine (Improvement #1
# uniqueness bonus + Improvement #3 auto-bind from commits a5/f12110f are
# UNCHANGED — both gating flags are read from config.plugin and the
# code path goes through the existing context_graph.ContextGraph
# implementation byte-for-byte).
#
# Edits vs graph_agent_isolated.py (4 lines total):
#   - import block: drop create_chat / *_SEARCH; add create_chat_code
#   - default workflow: "search_graph" -> "code_graph"
#   - branch_prompt / summary_prompt: drop search-vs-code dispatch
#   - env.close() at end of process_item
#
# VARIABLE / HYPOTHESIS node types from the design doc are NOT YET added —
# the basic OBSERVATION / SUBTASK / SUMMARY types already cover the paper
# claim (auto-bind cross-subtask edges + uniqueness across summaries).
# Phase D3 may revisit if early eval shows we need richer node typing.

from verl import DataProto
from .utils import Agent, select_env, truncate_text, is_weird, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics
from .finalizer import remaining_generation_tokens, step_preserving_final_answer
from .rollout_status import classify_rollout_status, validate_session_summary
from .prompts import BRANCH_MESSAGE
from .prompts_code import create_chat_code
from .verifier import judge_scope
from .context_graph import ContextGraph, GraphOpResult, NodeType, NodeStatus, EdgeRelation
from .graph_controller import (
    GraphActionController,
    GraphControllerError,
    graph_checkpoint_due,
)
from .graph_observation import record_tool_observation


from .utils import print_chat


from .agent_text import extract_fn_call, extract_summary, clean_response


# ── Graph operation handlers (parent-graph only in isolated variant) ──

from .graph_operations import handle_merge, handle_add_edge, handle_select, handle_prune, GRAPH_OPS





def make_graph_aware_run_action(env, child_graph: ContextGraph):
    """Wrap run_action so a branch's tool results also become OBSERVATION
    nodes in its private subgraph.

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
            record_tool_observation(child_graph, fn_call, observation)
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
    validate_session_summary(config.plugin)
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

    workflow = _get(item.non_tensor_batch['extra_info']).get('workflow', None) or getattr(config.plugin, "workflow", "code_graph")
    structured_graph_controller = bool(
        getattr(config.plugin, "structured_graph_controller", False)
    )
    user_prompt = create_chat_code(
        env.instance_info['problem_statement'],
        workflow,
        item,
        env=env,
        expose_graph_tools=not structured_graph_controller,
    )

    branch_prompt = BRANCH_MESSAGE

    max_turn = getattr(config.plugin, 'max_turn', 64) if config.plugin else 64
    max_session = getattr(config.plugin, "max_session", 5)
    if not is_train:
        max_session = getattr(config.plugin, "val_max_session", max_session)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    process_reward = getattr(config.plugin, "process_reward", None)
    if process_reward is not None and isinstance(process_reward, str) and process_reward.lower() == "none":
        process_reward = None
    max_traj = getattr(config.plugin, "max_traj", None)

    lambda_compact = getattr(config.plugin, "lambda_compact", 0.1)
    lambda_cost = getattr(config.plugin, "lambda_cost", 0.02)
    # Improvement #1: branch_uniqueness_bonus. Default 0 = off. When >0,
    # rewards branches whose summary content has low Jaccard overlap with
    # other sibling summaries. See context_graph._compute_branch_uniqueness.
    uniqueness_weight = getattr(config.plugin, "uniqueness_weight", 0.0)
    # Improvement #3: auto-bind branch summary to most-related existing node
    # with a SEMANTIC edge, in addition to the CAUSAL edge to its subtask.
    # Default False = off (preserves v3/v4 behavior). When True, branches
    # produce a graph rather than a tree (cross-subtask relations enabled).
    auto_bind_branch_edges = getattr(config.plugin, "auto_bind_branch_edges", False)
    auto_bind_min_overlap = getattr(config.plugin, "auto_bind_min_overlap", 0.05)
    auto_prune_max_active = max(
        0, int(getattr(config.plugin, "auto_prune_max_active", 12) or 0)
    )
    # Forced consolidation: every N main turns env injects a checkpoint where
    # policy MUST emit a graph op or <pass>. <pass> is reward-neutral only if
    # graph.is_saturated() returns True. 0 disables consolidation entirely
    # (back to v2 behavior).
    consolidation_interval = getattr(config.plugin, "consolidation_interval", 0)
    initial_consolidation_turn = getattr(
        config.plugin, "initial_consolidation_turn", 0
    )
    controller_action_policy = str(
        getattr(config.plugin, "controller_action_policy", "balanced")
    ).strip().lower()
    if controller_action_policy not in {"balanced", "structural"}:
        raise ValueError(
            "controller_action_policy must be balanced or structural; got "
            f"{controller_action_policy}"
        )
    # Valid graph mechanics are reward-neutral until final task success; SFT
    # teaches tool usage and the task-gated terminal reward supplies credit.
    consolidation_op_reward = getattr(config.plugin, "consolidation_op_reward", 0.0)
    consolidation_pass_invalid_penalty = getattr(
        config.plugin, "consolidation_pass_invalid_penalty", -0.2
    )
    consolidation_invalid_penalty = getattr(
        config.plugin, "consolidation_invalid_penalty", -0.3
    )

    llm_client = context.llm_client

    # ── Initialize parent ContextGraph ──
    graph = ContextGraph(tokenizer, namespace_prefix="n")
    query_text = env.instance_info['problem_statement']
    root_id = graph.add_node(query_text, NodeType.QUERY)
    graph_controller = GraphActionController(
        max_candidates=int(
            getattr(config.plugin, "graph_controller_max_candidates", 12)
        ),
        preview_chars=int(
            getattr(config.plugin, "graph_controller_preview_chars", 360)
        ),
        min_completion_tokens=int(
            getattr(
                config.plugin,
                "graph_controller_min_completion_tokens",
                256,
            )
        ),
    )
    branch_node_map = {}  # branch_name -> subtask_node_id
    branch_subgraph_stats = {}  # branch_name -> child graph stats (for logging)

    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    branches = []
    branch_tasks = {}
    branch_return = {}
    init_len = len(agent['main'].context())
    session_start_time = time.time()
    iteration = 0
    main_turn_count = 0   # counts only main-agent turns (not branch internals)
    controller_mode_rejections = 0
    consolidation_stats = {
        'attempts': 0, 'ops': 0, 'pass_valid': 0, 'pass_invalid': 0,
        'invalid': 0, 'budget_skips': 0, 'controller_errors': 0,
        'candidate_skips': 0,
    }
    mask_rollout = True
    timed_out = False
    session_message = []

    while iteration < max_turn:
        if time.time() - session_start_time > session_timeout:
            print('[SESSION] Session Timeout')
            timed_out = True
            break

        iteration += 1
        main_turn_count += 1
        # Tick the consolidation saturation clock at start of each main turn.
        # add_node() resets it back to 0 whenever new content arrives this turn.
        graph.turns_since_last_node_add += 1


        response = await agent['main'].step()

        if response is None:
            break

        session_message.append({'role': 'assistant', 'content': response})
        fn_call = extract_fn_call(response)

        # ── Graph operations on parent graph ──
        if (
            structured_graph_controller
            and fn_call is not None
            and fn_call['function'] in GRAPH_OPS
        ):
            controller_mode_rejections += 1
            observation = (
                "[CONTROLLER MODE REJECTION] Graph-management XML is not an "
                "environment tool. Continue with python_exec, branch, return, or "
                "finish. Graph changes are accepted only as controller-requested "
                "JSON inside [GRAPH ACTION MODE]."
            )
            print(
                f'[GRAPH CONTROLLER MODE REJECTION] '
                f'{fn_call["function"]} outside checkpoint'
            )

        elif fn_call is not None and fn_call['function'] in GRAPH_OPS:
            handler = {
                'merge': handle_merge,
                'add_edge': handle_add_edge,
                'select': handle_select,
                'prune': handle_prune,
            }[fn_call['function']]
            budget_error = graph.graph_op_budget_error()
            observation = (
                GraphOpResult(f"[Error] {budget_error}.\n\n{graph.to_state_text()}", False)
                if budget_error else handler(graph, fn_call)
            )
            graph.record_graph_op(observation.success)
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

                # Improvement #3: auto-bind branch summary to most-related EXISTING
                # parent node with a SEMANTIC edge. Without this, branches only
                # ever connect back to their own subtask (a tree, not a graph),
                # which is why edge/node ratio stayed flat at ~1.85 across v3
                # training. Use lexical Jaccard for matching — no embedding model
                # dependency. Bypasses self, subtask parent, and inactive nodes.
                if auto_bind_branch_edges:
                    summary_tok = set(branch_message[:2000].lower().split())
                    if summary_tok:
                        best_overlap, best_target = 0.0, None
                        for nid, node in graph.nodes.items():
                            if nid == summary_id or nid == subtask_id or not node.is_active():
                                continue
                            if node.type not in (NodeType.SUMMARY, NodeType.OBSERVATION):
                                continue
                            cand_tok = set(node.content.lower().split())
                            if not cand_tok:
                                continue
                            union = len(summary_tok | cand_tok)
                            overlap = len(summary_tok & cand_tok) / union if union else 0.0
                            if overlap > best_overlap:
                                best_overlap, best_target = overlap, nid
                        if best_target is not None and best_overlap > auto_bind_min_overlap:
                            graph.add_edge(
                                summary_id, best_target,
                                EdgeRelation.SEMANTIC, weight=best_overlap,
                            )
                            print(f'[BRANCH AUTO-BIND] {summary_id} -> {best_target} (Jaccard={best_overlap:.2f})')

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

            record_tool_observation(graph, fn_call, observation)

        # ── Auto graph operations on PARENT graph (only) ──
        # Parent graph stays small (subtask + summary + main observations),
        # so the same heuristic thresholds work fine.
        auto_pruned = (
            graph.auto_prune_low_value(max_active=auto_prune_max_active)
            if auto_prune_max_active > 0
            else []
        )
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

        # Match fold_agent.py: append observation in full (post truncate_text)
        # to the main chat. The c866e62 additions (graph auto-merge, per-tool
        # byte truncation, K=3 sliding window) are removed to align with the
        # paper's design — only branch/return + FoldGRPO should drive context
        # management, not a hard-coded sliding window.
        agent['main'].append({'role': 'user', 'content': observation})
        session_message.append({'role': 'user', 'content': observation})

        # ── Forced consolidation checkpoint ──
        # Every `consolidation_interval` main turns, inject a checkpoint
        # asking the policy to emit a graph op or <pass>. <pass> is valid
        # (reward-neutral) only when the graph is saturated; otherwise
        # penalized. See ContextGraph.is_saturated() for thresholds.
        checkpoint_due = graph_checkpoint_due(
            main_turn_count,
            max_turn,
            consolidation_interval,
            initial_consolidation_turn,
        )
        checkpoint_budget_error = (
            graph.graph_op_budget_error() if checkpoint_due else None
        )
        if checkpoint_due and checkpoint_budget_error:
            consolidation_stats['budget_skips'] += 1
            print(f'[CONSOL SKIP] {checkpoint_budget_error}')

        if (
            checkpoint_due
            and checkpoint_budget_error is None
            and structured_graph_controller
        ):
            candidate_snapshot = graph_controller.snapshot(graph)
            if not candidate_snapshot.candidates:
                consolidation_stats['candidate_skips'] += 1
                print('[GRAPH CONTROLLER SKIP] no legal graph candidates')
            else:
                allow_pass = graph.is_saturated()
                controller_prompt = graph_controller.action_prompt(
                    candidate_snapshot,
                    turn_id=main_turn_count,
                    allow_pass=allow_pass,
                    action_policy=controller_action_policy,
                )
                agent['main'].append({'role': 'user', 'content': controller_prompt})
                if not graph_controller.has_completion_budget(
                    remaining_generation_tokens(agent['main'])
                ):
                    agent['main'].rollback(k=1)
                    consolidation_stats['budget_skips'] += 1
                    print(
                        '[GRAPH CONTROLLER SKIP] insufficient completion '
                        'token budget'
                    )
                    continue
                session_message.append({'role': 'user', 'content': controller_prompt})

                controller_response = await step_preserving_final_answer(
                    agent['main'],
                    0,
                    completion_kwargs={
                        "structured_outputs": graph_controller.structured_outputs(
                            candidate_snapshot,
                            allow_pass=allow_pass,
                            action_policy=controller_action_policy,
                        ),
                        "sampling_params": {
                            "temperature": float(getattr(
                                config.plugin,
                                "graph_controller_temperature",
                                0.0,
                            )),
                            "top_p": float(getattr(
                                config.plugin,
                                "graph_controller_top_p",
                                1.0,
                            )),
                        },
                    },
                )
                if controller_response is None:
                    consolidation_stats['controller_errors'] += 1
                    print('[GRAPH CONTROLLER ERROR] structured completion returned no response')
                    break
                session_message.append({
                    'role': 'assistant', 'content': controller_response,
                })
                controller_turn_idx = len(agent['main'].chat) - 1
                consolidation_stats['attempts'] += 1
                iteration += 1
                try:
                    graph_call = graph_controller.resolve_action(
                        graph,
                        candidate_snapshot,
                        controller_response,
                        allow_pass=allow_pass,
                        action_policy=controller_action_policy,
                    )
                    controller_action = graph_call['function']
                    if controller_action == 'pass':
                        controller_observation = GraphOpResult(
                            'Pass accepted: graph is saturated.', True
                        )
                    else:
                        controller_observation = {
                            'merge': handle_merge,
                            'prune': handle_prune,
                            'add_edge': handle_add_edge,
                            'select': handle_select,
                        }[controller_action](graph, graph_call)
                        if not controller_observation.success:
                            raise GraphControllerError(
                                str(controller_observation).split("\n", 1)[0]
                            )
                except GraphControllerError as exc:
                    consolidation_stats['controller_errors'] += 1
                    if process_reward and is_train:
                        agent['main'].set_process_reward(
                            controller_turn_idx,
                            consolidation_invalid_penalty,
                        )
                    print(f'[GRAPH CONTROLLER ERROR] {exc}')
                    controller_ack = (
                        "[GRAPH ACTION REJECTED] The controller could not safely "
                        "apply this decision; continue the environment task."
                    )
                else:
                    if controller_action == 'pass':
                        consolidation_stats['pass_valid'] += 1
                    else:
                        graph.record_graph_op(True)
                        consolidation_stats['ops'] += 1
                    if (
                        controller_action != 'pass'
                        and process_reward
                        and is_train
                    ):
                        agent['main'].set_process_reward(
                            controller_turn_idx,
                            consolidation_op_reward,
                        )
                    print(
                        f'[GRAPH CONTROLLER {controller_action.upper()}] '
                        f'{controller_observation[:100]}'
                    )
                    controller_ack = (
                        f"[GRAPH ACTION APPLIED: {controller_action}] "
                        f"{controller_observation[:300]}"
                    )

                controller_ack = (
                    f"{controller_ack}\n\n[Latest ContextGraph state]\n"
                    f"{graph.to_state_text()}\n\n"
                    "[ENVIRONMENT MODE RESTORED] Continue the task normally. "
                    "Your next response must use only python_exec, branch, return, "
                    "or finish in the normal XML format."
                )
                agent['main'].append({'role': 'user', 'content': controller_ack})
                session_message.append({'role': 'user', 'content': controller_ack})

        if (
            checkpoint_due
            and checkpoint_budget_error is None
            and not structured_graph_controller
        ):
            is_sat = graph.is_saturated()
            n_active = len(graph.active_nodes)
            n_edges = len(graph.active_edges)
            n_nodes = len(graph.nodes)
            pass_clause = (
                "If no obvious merge/prune/edge improvement helps, emit "
                "<function=pass>{}</function>."
                if is_sat else
                "You MUST emit a real operation; <function=pass> is NOT valid here "
                "and will be penalized."
            )
            consol_prompt = (
                f"[CONSOLIDATION CHECKPOINT turn={main_turn_count}]\n"
                f"Current graph: {n_active} active nodes, {n_edges} edges, {n_nodes} total. "
                f"Choose ONE operation: merge / prune / add_edge / select / pass.\n"
                f"{pass_clause}\n"
                f"After this checkpoint you continue the task normally."
            )
            agent['main'].append({'role': 'user', 'content': consol_prompt})
            session_message.append({'role': 'user', 'content': consol_prompt})

            consol_response = await agent['main'].step()
            if consol_response is None:
                break
            session_message.append({'role': 'assistant', 'content': consol_response})
            consol_turn_idx = len(agent['main'].chat) - 1
            consol_fn = extract_fn_call(consol_response)
            consolidation_stats['attempts'] += 1
            iteration += 1  # account for the extra LLM step

            if consol_fn is not None and consol_fn['function'] in GRAPH_OPS:
                # Real op — apply the same handler as the main loop branch.
                handler = {
                    'merge': handle_merge,
                    'add_edge': handle_add_edge,
                    'select': handle_select,
                    'prune': handle_prune,
                }[consol_fn['function']]
                budget_error = graph.graph_op_budget_error()
                consol_obs = (
                    GraphOpResult(f"[Error] {budget_error}.\n\n{graph.to_state_text()}", False)
                    if budget_error else handler(graph, consol_fn)
                )
                graph.record_graph_op(consol_obs.success)
                if consol_obs.success:
                    consolidation_stats['ops'] += 1
                    if process_reward and is_train:
                        agent['main'].set_process_reward(consol_turn_idx, consolidation_op_reward)
                    print(f'[CONSOL OP] {consol_fn["function"]} -> {consol_obs[:100]}')
                    ack = f"[CONSOLIDATION OK] {consol_obs[:300]}"
                else:
                    consolidation_stats['invalid'] += 1
                    if process_reward and is_train:
                        agent['main'].set_process_reward(consol_turn_idx, consolidation_invalid_penalty)
                    print(f'[CONSOL INVALID] {consol_fn["function"]} -> {consol_obs[:100]}')
                    ack = f"[CONSOLIDATION INVALID] {consol_obs[:300]}"
            elif consol_fn is not None and consol_fn['function'] == 'pass':
                if is_sat:
                    consolidation_stats['pass_valid'] += 1
                    # reward-neutral
                    print('[CONSOL PASS] saturated, valid pass')
                    ack = "[CONSOLIDATION ACK] pass accepted (graph saturated)."
                else:
                    consolidation_stats['pass_invalid'] += 1
                    if process_reward and is_train:
                        agent['main'].set_process_reward(
                            consol_turn_idx, consolidation_pass_invalid_penalty)
                    print('[CONSOL PASS] NOT saturated -> penalty')
                    ack = "[CONSOLIDATION ACK] pass rejected (graph still sparse, op was expected)."
            else:
                consolidation_stats['invalid'] += 1
                if process_reward and is_train:
                    agent['main'].set_process_reward(
                        consol_turn_idx, consolidation_invalid_penalty)
                print('[CONSOL INVALID]', str(consol_response)[:120])
                ack = "[CONSOLIDATION ACK] invalid response, continuing."

            agent['main'].append({'role': 'user', 'content': ack})
            session_message.append({'role': 'user', 'content': ack})

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
        score[1],
        lambda_compact=lambda_compact,
        lambda_cost=lambda_cost,
        uniqueness_weight=uniqueness_weight,
    )
    print(f'[GRAPH REWARD ISOLATED] {graph_rewards} | branch_subgraphs={branch_subgraph_stats}')

    outs = []
    env.stats['get_final_score'] = score[1]
    env.stats['traj_num'] = len(agent)
    main_response_tokens = max(len(agent['main'].context()) - init_len, 0)
    main_context_tokens = len(agent['main'].context())
    working_context_limit = config.prompt_length + config.response_length
    env.stats['main_len'] = min(main_response_tokens, config.response_length)
    env.stats['main_context_tokens'] = main_context_tokens
    env.stats['working_context_limit'] = working_context_limit
    env.stats['total_token'] = len(tokenizer.encode(print_chat(user_prompt + session_message)))
    env.stats['main_turn'] = len(agent['main'].messages())
    env.stats['is_branch'] = int(len(agent) > 1)
    env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
    env.stats['use_all_branch'] = int(len(branches) + 1 > max_session)
    env.stats['graph_n_nodes'] = len(graph.nodes)
    env.stats['graph_n_active'] = len(graph.active_nodes)
    env.stats['graph_n_edges'] = len(graph.active_edges)
    env.stats['graph_ops'] = graph.operation_count
    env.stats['graph_op_attempts'] = graph.graph_op_attempt_count
    env.stats['graph_explicit_ops'] = graph.explicit_op_count
    env.stats['graph_invalid_ops'] = graph.invalid_op_count
    env.stats['graph_invalid_op_rate'] = (
        graph.invalid_op_count / graph.graph_op_attempt_count
        if graph.graph_op_attempt_count else 0.0
    )
    env.stats['controller_mode_rejections'] = controller_mode_rejections
    env.stats['controller_mode_rejection_rate'] = (
        controller_mode_rejections / main_turn_count if main_turn_count else 0.0
    )
    env.stats['graph_n_summaries'] = graph_rewards.get('n_summaries', 0)
    env.stats['graph_reward'] = graph_rewards.get('graph_reward', score[1])
    # Consolidation checkpoint stats — surface in wandb to track whether
    # the policy is actually using the forced-exploration channel.
    env.stats['consol_attempts'] = consolidation_stats['attempts']
    env.stats['consol_ops'] = consolidation_stats['ops']
    env.stats['consol_pass_valid'] = consolidation_stats['pass_valid']
    env.stats['consol_pass_invalid'] = consolidation_stats['pass_invalid']
    env.stats['consol_invalid'] = consolidation_stats['invalid']
    env.stats['consol_budget_skips'] = consolidation_stats['budget_skips']
    env.stats['consol_controller_errors'] = consolidation_stats['controller_errors']
    env.stats['consol_candidate_skips'] = consolidation_stats['candidate_skips']
    env.stats['structured_graph_controller'] = int(structured_graph_controller)
    env.stats['controller_structural_policy'] = int(
        controller_action_policy == "structural"
    )
    # Rates (denominator-safe; 0 when no consolidation fired)
    _ca = max(consolidation_stats['attempts'], 1)
    env.stats['consol_op_rate'] = consolidation_stats['ops'] / _ca
    env.stats['consol_valid_pass_rate'] = consolidation_stats['pass_valid'] / _ca
    env.stats['consol_invalid_pass_rate'] = consolidation_stats['pass_invalid'] / _ca
    env.stats['consol_invalid_rate'] = consolidation_stats['invalid'] / _ca
    # Pure task accuracy + isolated shaping component, exposed so wandb
    # can show CtxGraph's task hit-rate without graph_reward inflation.
    # task_reward should match score[1] (LocalSearch judge), but read from
    # graph_rewards dict for consistency with the breakdown.
    env.stats['task_reward'] = float(graph_rewards.get('task_reward', score[1]))
    env.stats['graph_shaping'] = float(graph_rewards.get('graph_shaping', 0.0))
    # Surface real-eval VER (valid_execution) + produced-file count so they
    # aggregate into val/* metrics. At 8B zero-shot SR floors to 0, so VER is
    # the signal that separates the agents. Only present under SAB_REAL_EVAL=1.
    if isinstance(reward_dict, dict):
        for _k in (
            'valid_execution', 'produced_files', 'valid_result_json',
            'hms_score', 'hms_context_recall', 'hms_mean_accuracy', 'judge_error',
        ):
            if _k in reward_dict:
                try:
                    env.stats[_k] = float(reward_dict[_k])
                except (TypeError, ValueError):
                    pass
    # Isolated-variant specific stats: aggregate child subgraph sizes
    env.stats['isolated_n_subgraphs'] = len(branch_subgraph_stats)
    env.stats['isolated_total_subgraph_nodes'] = sum(s.get('n_total', 0) for s in branch_subgraph_stats.values())
    env.stats['isolated_total_subgraph_obs'] = sum(s.get('n_observations', 0) for s in branch_subgraph_stats.values())

    if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
        mask_rollout = False
    if score[1] > 0:
        mask_rollout = False

    is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
    rollout_status = classify_rollout_status(
        response_tokens=main_response_tokens,
        response_limit=config.response_length,
        main_context_tokens=main_context_tokens,
        working_context_limit=working_context_limit,
        is_finish=is_finish,
        iteration=iteration,
        max_turn=max_turn,
        timed_out=timed_out,
    )
    env.stats.update({k: v for k, v in rollout_status.items() if k != 'termination_reason'})
    env.stats['concise_main'] = 1 - rollout_status['unfolded_main']
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
                'overlong': rollout_status['overlong'],
                'no_finish': rollout_status['no_finish'],
                'hit_token_limit': rollout_status['hit_token_limit'],
                'hit_max_turn': rollout_status['hit_max_turn'],
                'hit_timeout': rollout_status['hit_timeout'],
                'unfolded_main': rollout_status['unfolded_main'],
                'termination_reason': rollout_status['termination_reason'],
                'is_finish': is_finish,
                'agent_name': name,
                'message_str': print_chat(session_message),
                'meta_info': f"N: {len(agent)} | {name} | G:{len(graph.nodes)}n/{len(graph.active_edges)}e [iso]",
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

    # Release the per-trajectory sandbox namespace (ScienceAgentEnv).
    # LocalSearch.close() is a no-op (method may not exist).
    try:
        if hasattr(env, 'close'):
            env.close()
    except Exception:
        pass

    return outs
