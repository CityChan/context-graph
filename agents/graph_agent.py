"""ContextGraph Agent Loop: Graph-structured working context for LLM agents.

Extends FoldAgent's branch-based approach with a full context graph.
New capabilities over fold_agent:
- Cross-branch connections via add_edge
- Merge operations to consolidate findings
- Prune operations to discard dead ends
- Graph-aware reward computation (compactness, structural quality)
- Graph state tracking and visualization in observations

Compatible with verl training pipeline (same AgentLoopOutput format).
"""

from .rollout_status import validate_session_summary

import os
import time
import copy
import asyncio
from functools import partial
import random
from uuid import uuid4
from typing import Any, Union

from verl import DataProto
from .utils import Agent, select_env, truncate_text, is_weird, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics
from .prompts import create_chat, BRANCH_MESSAGE_SEARCH, BRANCH_MESSAGE
from .verifier import judge_scope
from .context_graph import ContextGraph, GraphOpResult, NodeType, NodeStatus, EdgeRelation


from .utils import print_chat


from .agent_text import extract_fn_call, extract_summary, clean_response


# ── Graph operation handlers ──

def handle_merge(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    """Handle merge tool call: combine nodes into summary."""
    node_ids_str = fn_call['arguments'].get('node_ids', '')
    summary = fn_call['arguments'].get('summary', '')
    node_ids = [nid.strip() for nid in node_ids_str.split(',') if nid.strip()]

    if len(node_ids) < 2:
        return GraphOpResult(f"[Error] merge requires at least 2 node IDs (got {len(node_ids)}).\n\n{graph.to_state_text()}", False)

    invalid = [nid for nid in node_ids if nid not in graph.nodes or not graph.nodes[nid].is_active()]
    if invalid:
        return GraphOpResult(f"[Error] Inactive or unknown node IDs: {invalid}.\n\n{graph.to_state_text()}", False)
    if len(set(node_ids)) != len(node_ids):
        return GraphOpResult(f"[Error] merge node IDs must be unique: {node_ids}.\n\n{graph.to_state_text()}", False)
    if len(node_ids) > 6:
        return GraphOpResult(f"[Error] merge accepts at most 6 node IDs (got {len(node_ids)}).\n\n{graph.to_state_text()}", False)
    if not summary.strip():
        return GraphOpResult(f"[Error] merge requires a non-empty summary.\n\n{graph.to_state_text()}", False)

    merged_id = graph.merge(node_ids, summary)
    if merged_id is None:
        return GraphOpResult(f"[Error] Could not merge nodes {node_ids}.\n\n{graph.to_state_text()}", False)

    return GraphOpResult(f"Merged {node_ids} into [{merged_id}].\n\n{graph.to_state_text()}", True)


def handle_add_edge(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    """Handle add_edge tool call: create relationship between nodes."""
    source = fn_call['arguments'].get('source', '').strip()
    target = fn_call['arguments'].get('target', '').strip()
    relation_str = fn_call['arguments'].get('relation', 'semantic').strip()

    relation_map = {
        'causal': EdgeRelation.CAUSAL,
        'semantic': EdgeRelation.SEMANTIC,
        'temporal': EdgeRelation.TEMPORAL,
    }
    relation = relation_map.get(relation_str, EdgeRelation.SEMANTIC)

    if source not in graph.nodes or not graph.nodes[source].is_active():
        return GraphOpResult(f"[Error] Inactive or unknown source node: {source}.\n\n{graph.to_state_text()}", False)
    if target not in graph.nodes or not graph.nodes[target].is_active():
        return GraphOpResult(f"[Error] Inactive or unknown target node: {target}.\n\n{graph.to_state_text()}", False)

    if not graph.add_edge(source, target, relation):
        return GraphOpResult(f"[Error] Cannot add self-loop or duplicate edge {source} --{relation_str}--> {target}.\n\n{graph.to_state_text()}", False)
    return GraphOpResult(f"Added edge {source} --{relation_str}--> {target}.\n\n{graph.to_state_text()}", True)


def handle_select(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    """Handle select tool call: change active focus."""
    node_id = fn_call['arguments'].get('node_id', '').strip()

    if node_id not in graph.nodes:
        return GraphOpResult(f"[Error] Unknown node: {node_id}.\n\n{graph.to_state_text()}", False)

    if not graph.select(node_id):
        return GraphOpResult(f"[Error] Cannot select {node_id} (inactive?).\n\n{graph.to_state_text()}", False)

    node = graph.nodes[node_id]
    content_preview = node.content[:500]
    return GraphOpResult(f"Focus shifted to [{node_id}] ({node.type.value}).\n\nContent:\n{content_preview}\n\n{graph.to_state_text()}", True)


def handle_prune(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    """Handle prune tool call: remove node from active context."""
    node_id = fn_call['arguments'].get('node_id', '').strip()

    if node_id not in graph.nodes:
        return GraphOpResult(f"[Error] Unknown node: {node_id}.\n\n{graph.to_state_text()}", False)

    if not graph.prune(node_id):
        return GraphOpResult(f"[Error] Cannot prune {node_id} (root or inactive node?).\n\n{graph.to_state_text()}", False)

    return GraphOpResult(f"Pruned [{node_id}].\n\n{graph.to_state_text()}", True)


GRAPH_OPS = {'merge', 'add_edge', 'select', 'prune'}


async def process_item(
        item: DataProto,
        context: TaskContext,
) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
    """ContextGraph agent loop: graph-structured context management.

    Extends fold_agent with:
    - ContextGraph tracking all observations, branches, and summaries
    - Graph manipulation tools (merge, add_edge, select, prune)
    - Graph-aware reward computation
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

    # Select env
    EnvClass = select_env(ability, config)
    print(is_train, EnvClass)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")

    # Create prompt (uses search_graph workflow)
    workflow = _get(item.non_tensor_batch['extra_info']).get('workflow', None) or getattr(config.plugin, "workflow", "search_graph")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)

    branch_prompt = BRANCH_MESSAGE_SEARCH if 'search' in workflow else BRANCH_MESSAGE

    max_turn = getattr(config.plugin, 'max_turn', 64) if config.plugin else 64
    max_session = getattr(config.plugin, "max_session", 5)
    if not is_train:
        max_session = getattr(config.plugin, "val_max_session", max_session)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    process_reward = getattr(config.plugin, "process_reward", None)
    if process_reward is not None and isinstance(process_reward, str) and process_reward.lower() == "none":
        process_reward = None
    max_traj = getattr(config.plugin, "max_traj", None)

    # Graph reward parameters
    lambda_compact = getattr(config.plugin, "lambda_compact", 0.1)
    lambda_cost = getattr(config.plugin, "lambda_cost", 0.02)

    llm_client = context.llm_client

    # ── Initialize ContextGraph ──
    graph = ContextGraph(tokenizer)
    query_text = env.instance_info['problem_statement']
    root_id = graph.add_node(query_text, NodeType.QUERY)
    # Track which graph nodes correspond to which branches
    branch_node_map = {}  # branch_name -> subtask_node_id

    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    branches = []
    branch_tasks = {}
    branch_return = {}
    init_len = len(agent['main'].context())
    session_start_time = time.time()
    iteration = 0
    mask_rollout = True
    session_message = []

    while iteration < max_turn:
        if time.time() - session_start_time > session_timeout:
            print('[SESSION] Session Timeout')
            break

        iteration += 1


        response = await agent['main'].step()

        if response is None:
            break

        session_message.append({'role': 'assistant', 'content': response})
        fn_call = extract_fn_call(response)

        # ── Handle graph operations ──
        if fn_call is not None and fn_call['function'] in GRAPH_OPS:
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
            print(f'[GRAPH] {fn_call["function"]} -> {observation[:100]}')

        # ── Handle branch (creates subtask node in graph) ──
        elif fn_call is not None and fn_call['function'] == 'branch':
            if len(branches) + 1 > max_session:
                observation = f"You've already reached the limit of {len(branches)} branch calls. Continue working independently."
            else:
                description = fn_call['arguments'].get('description', 'Agent')
                message_to_branch = fn_call['arguments'].get('prompt', 'Empty prompt')
                print('[BRANCH]', description, len(agent['main'].context()))

                # Create subtask node in graph
                subtask_id = graph.add_node(
                    f"Subtask: {description}\n{message_to_branch}",
                    NodeType.SUBTASK,
                    parent_id=graph.active_node_id,
                    edge_relation=EdgeRelation.DECOMPOSITION,
                )

                agent_name = f"#{len(branches)}-" + description.replace(' ', '_')
                branches.append(agent_name)
                branch_tasks[agent_name] = message_to_branch
                branch_node_map[agent_name] = subtask_id

                history = agent['main'].messages()
                agent[agent_name] = Agent(llm_client, history, tokenizer, config, prompt_turn=prompt_turn)
                branch_prompt_formatted = branch_prompt.format(message=message_to_branch)

                # Include graph state in branch observation prompt
                graph_state_hint = f"\n\nCurrent context graph:\n{graph.to_state_text()}"
                agent[agent_name].append({'role': 'user', 'content': branch_prompt_formatted + graph_state_hint})

                agent_return = await agent[agent_name].react(
                    partial(run_action, env),
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

                fn_call = extract_fn_call(last_response)
                branch_message = None
                if fn_call is not None and fn_call['function'] == 'return':
                    if 'message' in fn_call['arguments']:
                        branch_message = fn_call['arguments'].get('message', 'Empty message')
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                elif fn_call is not None and fn_call['function'] == 'finish':
                    if 'message' in fn_call['arguments']:
                        branch_message = fn_call['arguments'].get('message', 'Empty message')
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                if branch_message is None:
                    branch_message = f'Branch has finished its task. The last message was:\n\n{clean_response(last_response)}'

                # Add branch result as observation node connected to subtask
                obs_id = graph.add_node(
                    branch_message[:2000],
                    NodeType.OBSERVATION,
                    parent_id=subtask_id,
                    edge_relation=EdgeRelation.CAUSAL,
                )

                observation = f'{branch_message}\n\n{graph.to_state_text()}'
                branch_return[agent_name] = branch_message

        # ── Handle regular tools (search, open_page, finish) ──
        else:
            observation = await run_action(env, response)
            if observation is None:
                mask_rollout = False
                break

            # Track search/open_page results as observation nodes in graph
            if fn_call is not None:
                if fn_call['function'] == 'search':
                    obs_id = graph.add_node(
                        observation[:500],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.TEMPORAL,
                        metadata={'tool': 'search', 'query': fn_call['arguments'].get('query', '')},
                    )
                elif fn_call['function'] == 'open_page':
                    obs_id = graph.add_node(
                        observation[:800],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.CAUSAL,
                        metadata={'tool': 'open_page'},
                    )

        # ── Auto graph operations (heuristic, runs every turn) ──
        # max_active lowered from 20 to 12 to bound per-turn graph injection
        auto_pruned = graph.auto_prune_low_value(max_active=12)
        if auto_pruned:
            print(f'[GRAPH AUTO] Pruned low-value nodes: {auto_pruned}')

        auto_edges = graph.auto_connect_semantic(keyword_overlap_threshold=3)
        if auto_edges:
            print(f'[GRAPH AUTO] Added semantic edges: {auto_edges}')

        # Auto-merge: lowered threshold (3 -> 2) and now also fires when graph
        # exceeds soft cap (not only after branch returns), to keep node count
        # bounded for tasks with many shallow searches.
        should_try_merge = (
            (fn_call and fn_call.get('function') == 'branch')
            or len(graph.active_nodes) > 10
        )
        if should_try_merge:
            auto_merged = graph.auto_merge_similar(similarity_threshold=2)
            if auto_merged:
                print(f'[GRAPH AUTO] Merged observations into: {auto_merged}')

        if agent['main'].chat[-1]['role'] == 'user':
            print('[ROLE ERROR]')
            print(agent['main'].chat[-1])
            agent['main'].append({'role': 'assistant', 'content': str(response)})

        if process_reward:
            observation = truncate_text(observation, max_lines=100, merge_repeat=True, merge_num=4)

        agent['main'].append({'role': 'user', 'content': observation})
        session_message.append({'role': 'user', 'content': observation})

    env.stats['session_time'] = time.time() - session_start_time

    # ── Reward computation ──
    print('[TASK] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent['main'].messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    # ── Graph-aware reward ──
    graph_rewards = graph.compute_graph_reward(
        score[1], lambda_compact=lambda_compact, lambda_cost=lambda_cost
    )
    print(f'[GRAPH REWARD] {graph_rewards}')

    outs = []
    env.stats['get_final_score'] = score[1]
    env.stats['traj_num'] = len(agent)
    env.stats['main_len'] = min(len(agent['main'].context()) - init_len, config.response_length)
    env.stats['total_token'] = len(tokenizer.encode(print_chat(user_prompt + session_message)))
    env.stats['main_turn'] = len(agent['main'].messages())
    env.stats['is_branch'] = int(len(agent) > 1)
    env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
    env.stats['use_all_branch'] = int(len(branches) + 1 > max_session)
    # Graph-specific stats
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
    env.stats['graph_n_summaries'] = graph_rewards.get('n_summaries', 0)
    env.stats['graph_reward'] = graph_rewards.get('graph_reward', score[1])

    if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
        mask_rollout = False
    if score[1] > 0:
        mask_rollout = False

    is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
    if getattr(config.plugin, "must_finish", None):
        if not is_finish:
            score = ('', 0)

    # ── Process rewards (same as fold_agent + graph terms) ──
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
            # Penalize verbose main agent (exempt graph ops: merge/prune/select/add_edge)
            GRAPH_OP_MARKERS = ('<function=merge>', '<function=prune>', '<function=select>', '<function=add_edge>')
            if len(agent['main'].context()) - init_len > config.response_length * 0.5:
                bad_turn = [i for i, turn in enumerate(agent['main'].messages()) if
                            '<function=branch>' not in str(turn) and '<function=finish>' not in str(turn)
                            and not any(m in str(turn) for m in GRAPH_OP_MARKERS)]
                agent['main'].set_process_reward(bad_turn, -1)

            # Penalize no branching AND no graph ops (should use structure)
            has_graph_ops = graph.operation_count > len(graph.nodes)  # More ops than auto-adds
            if len(agent) == 1 and not has_graph_ops:
                agent['main'].set_process_reward('all', -1)
                if 'flat' in process_reward:
                    agent['main'].set_cache('reward', 1 - 1)

            # Per-turn graph operation rewards (conditioned on task success)
            if 'graph' in process_reward:
                for i, turn in enumerate(agent['main'].chat):
                    turn_str = str(turn)
                    # Reward merge turns: agent consolidated information effectively
                    if '<function=merge>' in turn_str:
                        agent['main'].set_process_reward(i, 0.2)
                    # Reward prune turns: agent cleaned up context
                    elif '<function=prune>' in turn_str:
                        agent['main'].set_process_reward(i, 0.1)
                    # Reward add_edge turns: agent made cross-branch connections
                    elif '<function=add_edge>' in turn_str:
                        agent['main'].set_process_reward(i, 0.1)
                    # Neutral for select (focus shift is tactical, not inherently good/bad)

            # Scope check
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

            # Tool call error
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

    # ── Build outputs ──
    # Use graph reward as the overall reward when graph process_reward is enabled
    use_graph_reward = process_reward and 'graph' in process_reward

    for name in agent if is_train else ['main']:
        out = await agent[name].get_data()
        agent_reward = score[1]

        # Reward priority: graph_reward > flat cache > raw task score
        # For main agent with graph enabled: use graph-modulated reward
        if use_graph_reward and name == 'main':
            agent_reward = graph_rewards.get('graph_reward', agent_reward)
        elif process_reward is not None and 'flat' in process_reward and 'reward' in agent[name].info_cache:
            # For branches or non-graph mode: use flat cache
            agent_reward = agent[name].info_cache['reward']

        # Always pass process_reward_mask (verl requires the key to exist).
        # If the mask is all zeros, verl's compute_foldgrpo_advantage will
        # detect this and skip the gmin/gmax patching path.
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
                'meta_info': f"N: {len(agent)} | {name} | G:{len(graph.nodes)}n/{len(graph.active_edges)}e",
                'process_reward_mask': out['process_reward_mask'],
                'uid': uid,
                'gen_uid': gen_uid,
                # Graph-specific fields
                'graph_state': graph.to_state_text(),
                'graph_rewards': graph_rewards,
            }
        )
        outs.append(copy.deepcopy(out))

    if max_traj is not None and len(outs) > max_traj:
        idx = [0] + sorted(random.sample(range(1, len(outs)), k=max_traj - 1))
        outs = [outs[i] for i in idx]

    return outs
