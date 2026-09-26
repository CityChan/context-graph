"""ContextGraph Agent Loop (Isolated Subgraph Variant).

Hierarchical/spatially-isolated version of context_graph_agent. Each branch
runs on its own private ContextGraph (a subgraph), so the main agent's
trajectory only ever sees:

  * The main (parent) graph: query root, subtask nodes, branch summary nodes,
    and the main agent's own search/open_page observations.
  * Only query-relevant branch evidence recovered through a bounded archive
    retrieval view, never the complete internal exploration trace.

When a branch returns, its child graph is collapsed into a SUMMARY node on
the parent and detached into an evidence archive. This bounds the main
agent's training sequence length to O(main_turns) instead of O(main_turns x
graph_size), eliminating the OOM that the global-injection variant runs into
on long-horizon tasks while retaining recoverability.

Trade-offs vs the global variant:
  + Main agent prompt size is bounded (≈ FoldAgent), no quadratic blow-up.
  + Each branch's subgraph is small and self-contained.
  + Cross-branch reasoning still possible at the parent level via add_edge
    between branch SUMMARY nodes.
  + Main agent normally reasons from summaries and can recover a small number
    of query-relevant branch observations when needed.
  - Branch agents currently do not invoke graph ops on their child graph
    (they only contribute observation nodes via the wrapped run_action).
    Adding branch-side graph ops is a follow-up.

This variant is registered as agent loop name ``context_graph_isolated_agent``.
"""

import os
import re
import time
import copy
import json
import asyncio
from functools import partial
import random
from uuid import uuid4
from typing import Any, Union

from verl import DataProto
from .utils import Agent, select_env, truncate_text, is_weird, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics
from .finalizer import (
    append_observation_preserving_final_answer,
    remaining_generation_tokens,
    step_preserving_final_answer,
    submit_emergency_final_answer,
)
from .rollout_status import classify_rollout_status
from .prompts import create_chat, BRANCH_MESSAGE_SEARCH, BRANCH_MESSAGE, SUMMARY_PROMPT_CODE, SUMMARY_PROMPT_SEARCH
from .verifier import judge_scope
from .context_graph import ContextGraph, GraphOpResult, NodeType, NodeStatus, EdgeRelation
from .graph_controller import (
    GraphActionController,
    GraphControllerError,
    graph_checkpoint_due,
)
from .graph_trace import GraphTraceRecorder
from .context_graph_modes import memory_mode, run_foldagent_equivalent
from .diagnostic_fixes import ANSWER_CONSISTENCY, RepeatAdvice, validate_fix
from .structured_memory import (
    GapStepScheduler,
    StructuredFactMemory,
    coerce_bool,
    fact_extraction_schema,
    gap_analysis_schema,
    plan_initialization_schema,
    parse_json_object,
    structured_memory_messages,
)
from .graph_rpo import (
    ANSWER_LIKELIHOOD_BACKENDS,
    EXTERNAL_EVALUATOR_BACKEND,
    OLD_POLICY_COUNTERFACTUAL_QA_BACKEND,
    GraphRPOEvaluatorError,
    assign_counterfactual_graph_edit_credits,
    assign_graph_edit_credits,
    format_counterfactual_qa_messages,
    graph_rpo_credit_backend,
    prepare_reference_graph_edit_requests,
)
from .trajectory_capture import serialize_agent_trajectories


_STRUCTURED_MEMORY_CONFIG_LOGGED = set()


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
    # vLLM returns None when token budget for a turn is negative (rollout skipped);
    # downstream code uses the cleaned text as a fallback branch summary, so emit
    # a clear placeholder rather than raising TypeError on None.
    if not response:
        return '[no response — turn skipped (token budget exhausted)]'
    if '<function=return>' in response:
        response = response.split('<function=return>')[-1]
    else:
        response = re.split(r'<\[[^\]]+\]>', response)[-1]
    return response


# ── Graph operation handlers (parent-graph only in isolated variant) ──

def handle_merge(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    node_ids_str = fn_call['arguments'].get('node_ids', '')
    summary = fn_call['arguments'].get('summary', '')
    node_ids = [nid.strip() for nid in node_ids_str.split(',') if nid.strip()]

    if len(node_ids) < 2:
        return GraphOpResult(f"[Error] merge requires at least 2 node IDs (got {len(node_ids)}).\n\n{graph.to_state_text()}", False)

    invalid = [
        nid for nid in node_ids
        if nid not in graph.nodes or not graph.nodes[nid].is_active()
    ]
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
    node_id = fn_call['arguments'].get('node_id', '').strip()

    if node_id not in graph.nodes:
        return GraphOpResult(f"[Error] Unknown node: {node_id}.\n\n{graph.to_state_text()}", False)

    if not graph.select(node_id):
        return GraphOpResult(f"[Error] Cannot select {node_id} (inactive?).\n\n{graph.to_state_text()}", False)

    node = graph.nodes[node_id]
    content_preview = node.content[:500]
    return GraphOpResult(f"Focus shifted to [{node_id}] ({node.type.value}).\n\nContent:\n{content_preview}\n\n{graph.to_state_text()}", True)


def handle_prune(graph: ContextGraph, fn_call: dict) -> GraphOpResult:
    node_id = fn_call['arguments'].get('node_id', '').strip()

    if node_id not in graph.nodes:
        return GraphOpResult(f"[Error] Unknown node: {node_id}.\n\n{graph.to_state_text()}", False)

    if not graph.prune(node_id):
        return GraphOpResult(f"[Error] Cannot prune {node_id} (root or inactive node?).\n\n{graph.to_state_text()}", False)

    return GraphOpResult(f"Pruned [{node_id}].\n\n{graph.to_state_text()}", True)


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
                        metadata={
                            'tool': 'search',
                            'query': fn_call['arguments'].get('query', ''),
                            'raw_content': observation,
                        },
                    )
                elif fn_call['function'] == 'open_page':
                    child_graph.add_node(
                        observation[:800],
                        NodeType.OBSERVATION,
                        parent_id=child_graph.active_node_id,
                        edge_relation=EdgeRelation.CAUSAL,
                        metadata={'tool': 'open_page', 'raw_content': observation},
                    )
                elif fn_call['function'] == 'think':
                    child_graph.add_node(
                        observation[:500],
                        NodeType.OBSERVATION,
                        parent_id=child_graph.active_node_id,
                        edge_relation=EdgeRelation.CAUSAL,
                        metadata={'tool': 'think', 'raw_content': observation},
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
         node and detached into a raw-evidence archive.
      3. During evaluation, old payloads leave the working chat and bounded
         retrieval restores relevant evidence. During training, chat history
         stays immutable so generated and optimized prompts remain identical.
    """
    mode = memory_mode(context.config.actor_rollout_ref.rollout.plugin)
    if mode == "foldagent":
        return await run_foldagent_equivalent(item, context)
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer

    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train
    diagnostic_fix = validate_fix(getattr(config.plugin, "diagnostic_fix", "none"), is_train)
    repeat_advice = RepeatAdvice()

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
    structured_graph_controller = bool(
        getattr(config.plugin, "structured_graph_controller", False)
    )
    user_prompt = create_chat(
        env.instance_info['problem_statement'],
        workflow,
        item,
        expose_graph_tools=not structured_graph_controller,
    )

    branch_prompt = BRANCH_MESSAGE_SEARCH if 'search' in workflow else BRANCH_MESSAGE
    summary_prompt = SUMMARY_PROMPT_SEARCH if 'search' in workflow else SUMMARY_PROMPT_CODE

    max_turn = getattr(config.plugin, 'max_turn', 64) if config.plugin else 64
    max_session = getattr(config.plugin, "max_session", 5)
    must_branch = bool(getattr(config.plugin, "must_branch", False))
    if not is_train:
        max_session = getattr(config.plugin, "val_max_session", max_session)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    process_reward = getattr(config.plugin, "process_reward", None)
    if process_reward is not None and isinstance(process_reward, str) and process_reward.lower() == "none":
        process_reward = None
    adv_estimator = str(getattr(context.config.algorithm, "adv_estimator", "")).lower()
    graph_rpo_enabled = adv_estimator in {"graphrpo", "advantageestimator.graphrpo"}
    graph_rpo_backend = graph_rpo_credit_backend(config.plugin) if graph_rpo_enabled else None
    if graph_rpo_enabled and is_train:
        if not structured_graph_controller:
            raise ValueError(
                "GraphRPO requires structured_graph_controller=True so each "
                "edit span is an isolated structured decision"
            )
        if (
            graph_rpo_backend == EXTERNAL_EVALUATOR_BACKEND
            and not str(getattr(config.plugin, "graph_rpo_evaluator_url", "") or "").strip()
        ):
            raise GraphRPOEvaluatorError(
                "GraphRPO requires actor_rollout_ref.rollout.plugin."
                "graph_rpo_evaluator_url"
            )
        if process_reward is None:
            process_reward = []
        elif isinstance(process_reward, str):
            process_reward = [process_reward]
        else:
            process_reward = list(process_reward)
        required_labels = ["graphrpo"]
        if bool(getattr(config.plugin, "graph_rpo_scope_process_reward", True)):
            required_labels.insert(0, "scope")
        for required_label in required_labels:
            if required_label not in process_reward:
                process_reward.append(required_label)
    max_traj = getattr(config.plugin, "max_traj", None)
    enable_summary = getattr(config.plugin, "enable_summary", False)
    enable_retrieval_memory = getattr(
        config.plugin, "enable_retrieval_memory", True
    )
    inject_graph_state_after_action = bool(
        getattr(config.plugin, "inject_graph_state_after_action", mode != "repaired")
    )
    controller_allow_pass = bool(
        getattr(config.plugin, "controller_allow_pass", False)
    )
    if mode == "repaired":
        controller_allow_pass = True
    # Rewriting a user turn after a later assistant turn has been generated
    # changes that assistant turn's conditioning context when VERL reconstructs
    # the training sequence. Keep training trajectories immutable. Evaluation
    # can still use in-place replacement because no log-prob is recomputed from
    # the completed transcript there.
    enable_history_replacement = enable_retrieval_memory and not is_train
    retrieval_summary_budget = getattr(
        config.plugin, "retrieval_summary_budget", 768
    )
    retrieval_evidence_budget = getattr(
        config.plugin, "retrieval_evidence_budget", 1280
    )
    retrieval_max_summaries = getattr(
        config.plugin, "retrieval_max_summaries", 5
    )
    retrieval_max_evidence = getattr(
        config.plugin, "retrieval_max_evidence", 4
    )
    working_memory_keep_recent = max(
        1, int(getattr(config.plugin, "working_memory_keep_recent", 1))
    )
    structured_memory_config_value = getattr(
        config.plugin, "structured_memory_enabled", None
    )
    structured_memory_env_value = os.getenv("STRUCTURED_MEMORY_ENABLED")
    structured_memory_requested = coerce_bool(
        structured_memory_config_value,
        default=coerce_bool(structured_memory_env_value, default=False),
    )
    structured_memory_required = coerce_bool(
        getattr(config.plugin, "structured_memory_required", None),
        default=coerce_bool(
            os.getenv("STRUCTURED_MEMORY_REQUIRED"), default=False
        ),
    )
    if structured_memory_required and not structured_memory_requested:
        raise ValueError(
            "structured memory was required by the launcher but resolved disabled; "
            f"plugin_value={structured_memory_config_value!r} "
            f"env_value={structured_memory_env_value!r}"
        )
    if structured_memory_requested and not structured_graph_controller:
        raise ValueError(
            "structured_memory_enabled requires structured_graph_controller=True "
            "so graph operations remain controller-owned and schema-constrained"
        )
    if structured_memory_requested and is_train:
        raise ValueError(
            "structured_memory_enabled is inference-only: auxiliary controller "
            "calls are not part of the optimized policy trajectory"
        )
    structured_memory_enabled = structured_memory_requested and not is_train
    structured_memory_config_key = (
        structured_memory_requested,
        structured_memory_required,
        is_train,
        repr(structured_memory_config_value),
        repr(structured_memory_env_value),
    )
    if structured_memory_config_key not in _STRUCTURED_MEMORY_CONFIG_LOGGED:
        _STRUCTURED_MEMORY_CONFIG_LOGGED.add(structured_memory_config_key)
        print(
            "[STRUCTURED MEMORY CONFIG] "
            f"plugin={structured_memory_config_value!r} "
            f"env={structured_memory_env_value!r} "
            f"required={structured_memory_required} "
            f"is_train={is_train} enabled={structured_memory_enabled}"
        )
    structured_memory_tools = {
        tool.strip()
        for tool in str(getattr(
            config.plugin,
            "structured_memory_tools",
            "search,open_page,branch_return",
        )).split(",")
        if tool.strip()
    }
    structured_memory_gap_interval = max(
        0, int(getattr(config.plugin, "structured_memory_gap_interval", 8) or 0)
    )
    structured_memory_context_budget = max(
        128, int(getattr(config.plugin, "structured_memory_context_budget", 1024))
    )
    structured_memory_max_context_facts = max(
        1, int(getattr(config.plugin, "structured_memory_max_context_facts", 12))
    )
    structured_memory_max_facts_per_observation = max(
        1, int(getattr(
            config.plugin, "structured_memory_max_facts_per_observation", 8
        ))
    )
    structured_memory_extract_max_tokens = max(
        128, int(getattr(config.plugin, "structured_memory_extract_max_tokens", 768))
    )
    structured_memory_gap_max_tokens = max(
        128, int(getattr(config.plugin, "structured_memory_gap_max_tokens", 512))
    )
    structured_memory_plan_max_tokens = max(
        128, int(getattr(config.plugin, "structured_memory_plan_max_tokens", 512))
    )
    structured_memory_relation_candidates = max(
        1, int(getattr(
            config.plugin, "structured_memory_relation_candidates", 128
        ))
    )
    structured_memory_controller_retries = max(
        0, int(getattr(config.plugin, "structured_memory_controller_retries", 2))
    )
    structured_memory_gap_jitter = max(
        0, int(getattr(config.plugin, "structured_memory_gap_jitter", 1))
    )
    structured_memory_stop_on_ready = coerce_bool(
        getattr(config.plugin, "structured_memory_stop_on_ready", None),
        default=True,
    )
    structured_memory_step_limit = max(
        0, int(getattr(config.plugin, "structured_memory_step_limit", 40) or 0)
    )
    effective_max_turn = (
        min(max_turn, structured_memory_step_limit)
        if structured_memory_enabled and structured_memory_step_limit
        else max_turn
    )

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
    # Keep the historical cap by default, but allow retention-focused evals to
    # postpone heuristic pruning.  A value of 0 disables auto-pruning while
    # leaving explicit controller/model prune operations available.
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
    if mode == "repaired":
        controller_action_policy = "balanced"
    if controller_action_policy not in {"balanced", "structural"}:
        raise ValueError(
            "controller_action_policy must be balanced or structural; got "
            f"{controller_action_policy}"
        )
    final_answer_reserve = max(
        int(getattr(config.plugin, "final_answer_reserve", 0) or 0), 0
    )
    final_answer_safety_margin = max(
        int(getattr(config.plugin, "final_answer_safety_margin", 64) or 0), 0
    )
    protected_final_answer_budget = (
        final_answer_reserve + final_answer_safety_margin
        if final_answer_reserve else 0
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
    graph_invalid_penalty = getattr(
        config.plugin, "graph_invalid_penalty", -0.3
    )
    if graph_rpo_enabled:
        # Section 4.2 assigns every invalid or malformed graph decision,
        # including a disallowed pass, the same -0.3 process label.
        consolidation_invalid_penalty = -0.3
        consolidation_pass_invalid_penalty = -0.3
        graph_invalid_penalty = -0.3

    llm_client = context.llm_client

    # ── Initialize parent ContextGraph ──
    graph = ContextGraph(tokenizer, namespace_prefix="n", memory_policy=mode)
    query_text = env.instance_info['problem_statement']
    root_id = graph.add_node(query_text, NodeType.QUERY)
    structured_memory = (
        StructuredFactMemory(
            query_text,
            tokenizer=tokenizer,
            max_facts=max(
                1, int(getattr(config.plugin, "structured_memory_max_facts", 128))
            ),
        )
        if structured_memory_enabled else None
    )
    structured_memory_stats = {
        "plan_calls": 0,
        "plan_errors": 0,
        "extraction_calls": 0,
        "extraction_errors": 0,
        "facts_added": 0,
        "facts_deduplicated": 0,
        "links_added": 0,
        "semantic_updates": 0,
        "zero_fact_extractions": 0,
        "gap_calls": 0,
        "gap_errors": 0,
        "gap_fallbacks": 0,
        "controller_retries": 0,
        "early_stops": 0,
        "context_injections": 0,
        "context_injection_tokens": 0,
        "max_context_snapshot_tokens": 0,
        "persistent_context_injections": 0,
    }
    graph_trace = GraphTraceRecorder(graph)
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

    if diagnostic_fix == "answer":
        user_prompt = copy.deepcopy(user_prompt)
        user_prompt[-1]['content'] += "\n\n" + ANSWER_CONSISTENCY
    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = Agent(
        llm_client,
        user_prompt,
        tokenizer,
        config,
        prompt_turn=prompt_turn,
        process_reward_min_precedence=graph_rpo_enabled,
    )

    async def call_structured_memory_controller(prompt, schema, max_new_tokens):
        """Run an isolated controller call with bounded schema-repair retries."""
        required = set(schema.get("required", []))
        previous_response = ""
        for attempt in range(structured_memory_controller_retries + 1):
            # Agent action turns are deliberately capped at 512 tokens in the
            # ALFWorld launcher.  Structured JSON may need more room to close
            # its arrays/object, especially after several facts accumulate.
            # Let isolated controller calls use their own budget and expand it
            # only after a failed parse; otherwise deterministic retries merely
            # reproduce the same truncated prefix.
            attempt_max_tokens = max_new_tokens * (attempt + 1)
            attempt_prompt = prompt
            if attempt:
                structured_memory_stats["controller_retries"] += 1
                attempt_prompt = (
                    "Your previous response was invalid. Re-emit the decision as one "
                    "complete JSON object satisfying every required key. The prior "
                    "response may have been truncated: use only necessary array items, "
                    "keep every string concise, and close the object. Do not explain the "
                    "correction.\n\nORIGINAL REQUEST:\n"
                    f"{prompt}\n\nINVALID RESPONSE:\n{previous_response[:2000]}"
                )
            messages = structured_memory_messages(attempt_prompt, schema)
            controller = Agent(
                llm_client,
                messages,
                tokenizer,
                config,
                prompt_turn=len(messages),
                chat_template_kwargs={"enable_thinking": False},
            )
            response = await controller.step(
                max_new_tokens=attempt_max_tokens,
                completion_kwargs={
                    "bypass_turn_max_new_tokens": True,
                    "structured_outputs": {"json": schema},
                    "sampling_params": {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_tokens": attempt_max_tokens,
                    },
                },
            )
            previous_response = response or ""
            payload = parse_json_object(previous_response)
            if payload is not None and required.issubset(payload):
                return payload
        return None

    gap_scheduler = GapStepScheduler(
        structured_memory_gap_interval,
        structured_memory_gap_jitter,
        query_text,
    )
    last_gap_revision = -1
    last_stall_gap_revision = -1
    no_progress_streak = 0

    async def refresh_structured_memory_gaps(reason):
        """Refresh K_cov/K_mis/Q_sug, falling back conservatively on failure."""
        nonlocal last_gap_revision
        if structured_memory is None:
            return
        structured_memory_stats["gap_calls"] += 1
        try:
            payload = await call_structured_memory_controller(
                structured_memory.gap_prompt(),
                gap_analysis_schema(),
                structured_memory_gap_max_tokens,
            )
            if payload is None:
                raise ValueError("gap analyzer returned no JSON object")
        except Exception as error:
            structured_memory_stats["gap_errors"] += 1
            print(f"[STRUCTURED MEMORY ERROR] gap analysis failed: {error}")
            payload = structured_memory.fallback_gap_analysis()
            structured_memory_stats["gap_fallbacks"] += 1
            print("[STRUCTURED MEMORY GAP FALLBACK] conservative guidance applied")
        structured_memory.update_gaps(payload)
        last_gap_revision = structured_memory.revision
        print(
            "[STRUCTURED MEMORY GAP] "
            f"reason={reason} can_answer={structured_memory.can_answer} "
            f"gaps={len(structured_memory.missing_information)}"
        )

    def structured_memory_overlay(query, *, final=False):
        """Build and account for one transient, non-persistent snapshot."""
        if structured_memory is None:
            return None
        if final:
            snapshot = structured_memory.render_answer_context(
                max_facts=max(structured_memory_max_context_facts, 24),
                max_tokens=structured_memory_context_budget,
            )
        else:
            snapshot = structured_memory.render_context(
                query,
                max_facts=structured_memory_max_context_facts,
                max_tokens=structured_memory_context_budget,
            )
        if not snapshot:
            return None
        try:
            snapshot_tokens = len(tokenizer.encode(
                snapshot, add_special_tokens=False
            ))
        except TypeError:
            snapshot_tokens = len(tokenizer.encode(snapshot))
        structured_memory_stats["context_injections"] += 1
        structured_memory_stats["context_injection_tokens"] += snapshot_tokens
        structured_memory_stats["max_context_snapshot_tokens"] = max(
            structured_memory_stats["max_context_snapshot_tokens"],
            snapshot_tokens,
        )
        return snapshot

    async def update_structured_memory(
        observation_text,
        evidence_node_id,
        tool_name,
    ):
        """Extract and integrate facts from one evidence-bearing observation."""
        nonlocal no_progress_streak
        if (
            structured_memory is None
            or not evidence_node_id
        ):
            return
        node = graph.nodes.get(evidence_node_id)
        if node is None:
            return
        structured_memory.record_observation(
            evidence_node_id=evidence_node_id,
            tool_name=tool_name,
            observation=observation_text,
            source_metadata=node.metadata,
        )
        if tool_name not in structured_memory_tools:
            return
        structured_memory_stats["extraction_calls"] += 1
        try:
            prompt = structured_memory.extraction_prompt(
                observation_text,
                evidence_node_id=evidence_node_id,
                source_metadata=node.metadata,
                max_existing=structured_memory_relation_candidates,
            )
            payload = await call_structured_memory_controller(
                prompt,
                fact_extraction_schema(
                    structured_memory_max_facts_per_observation
                ),
                structured_memory_extract_max_tokens,
            )
            if payload is None:
                raise ValueError("fact extractor returned no JSON object")
            result = structured_memory.add_extraction(
                payload,
                evidence_node_id=evidence_node_id,
                source_metadata=node.metadata,
            )
            structured_memory_stats["facts_added"] += result["added"]
            structured_memory_stats["facts_deduplicated"] += result["deduplicated"]
            structured_memory_stats["links_added"] += result["links_added"]
            if result["changed"]:
                structured_memory_stats["semantic_updates"] += 1
                no_progress_streak = 0
            else:
                structured_memory_stats["zero_fact_extractions"] += 1
                no_progress_streak += 1
            print(
                "[STRUCTURED MEMORY] "
                f"tool={tool_name} node={evidence_node_id} "
                f"+{result['added']} facts +{result['links_added']} links "
                f"total={result['total_facts']} revision={result['revision']}"
            )
        except Exception as error:
            structured_memory_stats["extraction_errors"] += 1
            structured_memory_stats["zero_fact_extractions"] += 1
            no_progress_streak += 1
            print(f"[STRUCTURED MEMORY ERROR] extraction failed: {error}")

    async def maybe_refresh_structured_memory_gaps(turn):
        """Run paper-style step scheduling without retrying one stalled revision."""
        nonlocal last_stall_gap_revision, no_progress_streak
        if structured_memory is None:
            return
        periodic_gap_due = gap_scheduler.due(turn)
        stalled_gap_due = bool(
            no_progress_streak >= 2
            and structured_memory.revision != last_stall_gap_revision
        )
        if not periodic_gap_due and not stalled_gap_due:
            return
        if periodic_gap_due:
            gap_scheduler.mark_attempt(turn)
            reason = "periodic-step"
        else:
            last_stall_gap_revision = structured_memory.revision
            reason = "no-progress"
        await refresh_structured_memory_gaps(reason)
        no_progress_streak = 0

    if structured_memory is not None:
        structured_memory_stats["plan_calls"] += 1
        try:
            plan_payload = await call_structured_memory_controller(
                structured_memory.plan_prompt(),
                plan_initialization_schema(),
                structured_memory_plan_max_tokens,
            )
            if plan_payload is None:
                raise ValueError("plan initializer returned no JSON object")
            structured_memory.initialize_plan(plan_payload)
        except Exception as error:
            structured_memory_stats["plan_errors"] += 1
            structured_memory.initialize_plan({})
            print(f"[STRUCTURED MEMORY ERROR] plan initialization failed: {error}")
    branches = []
    branch_tasks = {}
    branch_return = {}
    init_len = len(agent['main'].context())
    current = 'main'
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
    natural_finish = False
    pre_finalize_token_limit = False
    observation_budget_truncations = 0
    observation_budget_skips = 0
    session_message = []
    working_memory_turns = []
    retrieval_totals = {
        'calls': 0,
        'summary_tokens': 0,
        'evidence_tokens': 0,
        'max_context_tokens': 0,
    }

    while iteration < effective_max_turn:
        if time.time() - session_start_time > session_timeout:
            print('[SESSION] Session Timeout')
            timed_out = True
            break

        if (
            structured_memory is not None
            and structured_memory_stop_on_ready
            and structured_memory.ready_to_answer
            and final_answer_reserve
        ):
            structured_memory_stats["early_stops"] += 1
            print(
                "[STRUCTURED MEMORY READY] all goals covered; "
                "stopping exploration for final synthesis"
            )
            break

        iteration += 1
        main_turn_count += 1
        # Tick the consolidation saturation clock at start of each main turn.
        # add_node() resets it back to 0 whenever new content arrives this turn.
        graph.turns_since_last_node_add += 1

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
            trace_before = graph_trace.capture(graph)
            graph.add_node(summary, NodeType.SUMMARY,
                          parent_id=graph.active_node_id,
                          edge_relation=EdgeRelation.TEMPORAL)
            graph_trace.record(
                graph, trace_before, turn_id=main_turn_count,
                source="system", op="session_summary",
                args={"summary": summary}, success=True,
                assistant_content=response,
            )
            next_session_prompt = (
                f"For this question, you have already made the following progress in previous session, "
                f"summarized as follow:\n\n{summary}\n\nNow continue work on it.")
            current = current + '+'
            agent[current] = Agent(
                llm_client,
                user_prompt,
                tokenizer,
                config,
                prompt_turn=prompt_turn,
                process_reward_min_precedence=graph_rpo_enabled,
            )
            agent[current].append({'role': 'assistant', 'content': ""})
            agent[current].append({'role': 'user', 'content': next_session_prompt})
            session_message.append({'role': 'user', 'content': next_session_prompt})

        memory_overlay = structured_memory_overlay(
            agent['main'].messages()[-1].get("content", query_text)
            if agent['main'].messages() else query_text
        )
        response = await step_preserving_final_answer(
            agent['main'],
            protected_final_answer_budget,
            context_overlay=memory_overlay,
        )

        if response is None:
            pre_finalize_token_limit = bool(
                final_answer_reserve
                and remaining_generation_tokens(agent['main'])
                <= protected_final_answer_budget + 9
            )
            break

        session_message.append({'role': 'assistant', 'content': response})
        fn_call = extract_fn_call(response)
        repeat_hint = ""
        if diagnostic_fix == "repeat" and fn_call:
            repeat_hint = repeat_advice.observe(fn_call['function'], fn_call.get('arguments', {}))
        new_evidence_node_id = None
        new_evidence_tool_name = None
        new_evidence_text = None

        # ── Graph operations on parent graph ──
        if (
            structured_graph_controller
            and fn_call is not None
            and fn_call['function'] in GRAPH_OPS
        ):
            controller_mode_rejections += 1
            trace_before = graph_trace.capture(graph)
            observation = (
                "[CONTROLLER MODE REJECTION] Graph-management XML is not an "
                "environment tool. Continue with search, open_page, branch, or "
                "finish. Graph changes are accepted only as controller-requested "
                "JSON inside [GRAPH ACTION MODE]."
            )
            graph_trace.record(
                graph, trace_before, turn_id=main_turn_count,
                source="model", op="controller_mode_rejection",
                args=fn_call.get('arguments', {}), success=False,
                error=f"out-of-mode graph action: {fn_call['function']}",
                assistant_content=response,
                assistant_turn_index=len(agent['main'].chat) - 1,
                decision_context={
                    "mode": "environment",
                    "rejected_op": fn_call['function'],
                },
            )
            if process_reward and is_train:
                agent['main'].set_process_reward(
                    len(agent['main'].chat) - 1, graph_invalid_penalty
                )
            print(
                f'[GRAPH CONTROLLER MODE REJECTION] '
                f'{fn_call["function"]} outside checkpoint'
            )

        # 鈹€鈹€ Graph operations on parent graph 鈹€鈹€
        elif fn_call is not None and fn_call['function'] in GRAPH_OPS:
            handler = {
                'merge': handle_merge,
                'add_edge': handle_add_edge,
                'select': handle_select,
                'prune': handle_prune,
            }[fn_call['function']]
            trace_before = graph_trace.capture(graph)
            budget_error = graph.graph_op_budget_error()
            observation = (
                GraphOpResult(f"[Error] {budget_error}.\n\n{graph.to_state_text()}", False)
                if budget_error else handler(graph, fn_call)
            )
            graph.record_graph_op(observation.success)
            graph_trace.record(
                graph, trace_before, turn_id=main_turn_count,
                source="model", op=fn_call['function'],
                args=fn_call.get('arguments', {}), success=observation.success,
                error=None if observation.success else str(observation).split("\n", 1)[0],
                assistant_content=response,
                assistant_turn_index=len(agent['main'].chat) - 1,
            )
            if not observation.success and process_reward and is_train:
                graph_turn_idx = len(agent['main'].chat) - 1
                agent['main'].set_process_reward(
                    graph_turn_idx, graph_invalid_penalty
                )
            print(f'[GRAPH ISOLATED] {fn_call["function"]} -> {observation[:100]}')

        # ``pass`` is only useful when the graph has no productive operation.
        # Exposing and handling it explicitly keeps consolidation checkpoints
        # inside the same XML tool protocol as every other graph action.
        elif fn_call is not None and fn_call['function'] == 'pass':
            trace_before = graph_trace.capture(graph)
            if graph.is_saturated():
                observation = GraphOpResult(
                    f"[Graph] pass accepted; no consolidation needed.\n\n{graph.to_state_text()}",
                    True,
                )
            else:
                observation = GraphOpResult(
                    f"[Error] pass rejected; graph still has productive operations.\n\n{graph.to_state_text()}",
                    False,
                )
                if process_reward and is_train:
                    pass_turn_idx = len(agent['main'].chat) - 1
                    agent['main'].set_process_reward(
                        pass_turn_idx, consolidation_pass_invalid_penalty
                    )
            graph_trace.record(
                graph, trace_before, turn_id=main_turn_count,
                source="model", op="pass", args={}, success=observation.success,
                error=None if observation.success else str(observation).split("\n", 1)[0],
                assistant_content=response,
                assistant_turn_index=len(agent['main'].chat) - 1,
            )

        # ── Branch: spawn isolated child subgraph ──
        elif fn_call is not None and fn_call['function'] == 'branch':
            trace_before = graph_trace.capture(graph)
            if len(branches) + 1 > max_session:
                observation = f"You've already reached the limit of {len(branches)} branch calls. Continue working independently."
                graph_trace.record(
                    graph, trace_before, turn_id=main_turn_count,
                    source="model", op="branch",
                    args=fn_call.get('arguments', {}), success=False,
                    error="branch session limit reached", assistant_content=response,
                )
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
                agent[agent_name] = Agent(
                    llm_client,
                    history,
                    tokenizer,
                    config,
                    prompt_turn=prompt_turn,
                    process_reward_min_precedence=graph_rpo_enabled,
                )
                branch_prompt_formatted = branch_prompt.format(message=message_to_branch)

                # Branch sees its OWN child graph state, not the parent
                graph_state_hint = f"\n\nYour subtask working graph:\n{child_graph.to_state_text()}"
                agent[agent_name].append({'role': 'user', 'content': branch_prompt_formatted + graph_state_hint})

                # 3. Branch runs with a graph-aware run_action that tracks
                #    its observations into the child graph (not the parent)
                wrapped_action = make_graph_aware_run_action(env, child_graph)
                agent_return = await agent[agent_name].react(
                    wrapped_action,
                    max_turn=max(1, effective_max_turn - iteration),
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
                #    subgraph's stats become metadata. The child is detached
                #    from active state and optionally retained in the evidence
                #    archive; it is never serialized wholesale into prompts.
                child_stats = graph.collapse_child(
                    subtask_id,
                    preserve_archive=enable_retrieval_memory,
                ) or {}
                branch_subgraph_stats[agent_name] = child_stats

                summary_id = graph.add_node(
                    branch_message if mode == "repaired" else branch_message[:2000],
                    NodeType.SUMMARY,
                    parent_id=subtask_id,
                    edge_relation=EdgeRelation.CAUSAL,
                    metadata={
                        'tool': 'branch_return',
                        'collapsed_from_branch': agent_name,
                        'child_graph_prefix': child_prefix,
                        **{f'child_{k}': v for k, v in child_stats.items()},
                    },
                )
                archive_id = child_stats.get('archive_id')
                if archive_id:
                    graph.attach_archive(summary_id, archive_id)
                new_evidence_node_id = summary_id
                new_evidence_tool_name = "branch_return"
                new_evidence_text = branch_message
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
                graph_trace.record(
                    graph, trace_before, turn_id=main_turn_count,
                    source="model", op="branch",
                    args=fn_call.get('arguments', {}), success=True,
                    assistant_content=response,
                )

        # ── Regular tools on main agent: add observation to PARENT graph ──
        elif (
            fn_call is not None
            and fn_call['function'] == 'finish'
            and must_branch
            and not branches
        ):
            observation = (
                "[ContextGraph requirement] Create at least one branch for an "
                "independent sub-calculation before submitting the final answer."
            )

        else:
            trace_before = graph_trace.capture(graph)
            observation = await run_action(env, response)
            if observation is None:
                mask_rollout = False
                natural_finish = True
                break

            if fn_call is not None:
                if fn_call['function'] == 'search':
                    new_evidence_node_id = graph.add_node(
                        observation[:500],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.TEMPORAL,
                        metadata={
                            'tool': 'search',
                            'query': fn_call['arguments'].get('query', ''),
                            'raw_content': observation,
                        },
                    )
                    new_evidence_tool_name = "search"
                    new_evidence_text = observation
                elif fn_call['function'] == 'open_page':
                    page_metadata = {
                        key: str(fn_call['arguments'].get(key, ''))[:500]
                        for key in ('url', 'docid', 'page', 'cursor')
                        if fn_call['arguments'].get(key) is not None
                    }
                    new_evidence_node_id = graph.add_node(
                        observation[:800],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.CAUSAL,
                        metadata={
                            'tool': 'open_page',
                            **page_metadata,
                            'raw_content': observation,
                        },
                    )
                    new_evidence_tool_name = "open_page"
                    new_evidence_text = observation
                elif fn_call['function'] == 'action':
                    new_evidence_node_id = graph.add_node(
                        observation[:300],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.TEMPORAL,
                        metadata={
                            'tool': 'action',
                            'command': fn_call['arguments'].get('command', '')[:100],
                            'raw_content': observation,
                        },
                    )
                    new_evidence_tool_name = "action"
                    new_evidence_text = observation
                elif fn_call['function'] == 'think':
                    new_evidence_node_id = graph.add_node(
                        observation[:500],
                        NodeType.OBSERVATION,
                        parent_id=graph.active_node_id,
                        edge_relation=EdgeRelation.CAUSAL,
                        metadata={'tool': 'think', 'raw_content': observation},
                    )
                    new_evidence_tool_name = "think"
                    new_evidence_text = observation
            if new_evidence_node_id is not None:
                graph_trace.record(
                    graph, trace_before, turn_id=main_turn_count,
                    source="environment", op="add_observation",
                    args={
                        "tool": fn_call.get('function') if fn_call else None,
                        "node_id": new_evidence_node_id,
                    },
                    success=True,
                    assistant_content=response,
                )

        # ── Auto graph operations on PARENT graph (only) ──
        if new_evidence_node_id is not None:
            await update_structured_memory(
                new_evidence_text if new_evidence_text is not None else observation,
                new_evidence_node_id,
                new_evidence_tool_name or "unknown",
            )
        await maybe_refresh_structured_memory_gaps(main_turn_count)

        # Parent graph stays small (subtask + summary + main observations),
        # so the same heuristic thresholds work fine.
        trace_before = graph_trace.capture(graph)
        auto_pruned = (
            graph.auto_prune_low_value(max_active=auto_prune_max_active)
            if auto_prune_max_active > 0
            else []
        )
        if auto_pruned:
            graph_trace.record(
                graph, trace_before, turn_id=main_turn_count,
                source="heuristic", op="auto_prune",
                args={"node_ids": auto_pruned}, success=True,
            )
            print(f'[GRAPH ISOLATED AUTO] Pruned low-value nodes: {auto_pruned}')

        trace_before = graph_trace.capture(graph)
        auto_edges = graph.auto_connect_semantic(keyword_overlap_threshold=3)
        if auto_edges:
            graph_trace.record(
                graph, trace_before, turn_id=main_turn_count,
                source="heuristic", op="auto_connect",
                args={"edges": auto_edges}, success=True,
            )
            print(f'[GRAPH ISOLATED AUTO] Added semantic edges: {auto_edges}')

        if fn_call and fn_call.get('function') == 'branch':
            trace_before = graph_trace.capture(graph)
            auto_merged = graph.auto_merge_similar(similarity_threshold=2)
            if auto_merged:
                graph_trace.record(
                    graph, trace_before, turn_id=main_turn_count,
                    source="heuristic", op="auto_merge",
                    args={"summary_node_id": auto_merged}, success=True,
                )
                print(f'[GRAPH ISOLATED AUTO] Merged: {auto_merged}')

        if agent['main'].chat[-1]['role'] == 'user':
            print('[ROLE ERROR]')
            print(agent['main'].chat[-1])
            agent['main'].append({'role': 'assistant', 'content': str(response)})

        if process_reward:
            observation = truncate_text(observation, max_lines=100, merge_repeat=True, merge_num=4)
        if repeat_hint:
            observation = repeat_hint + "\n\n" + observation

        # The newest payload remains verbatim for immediate reasoning. Older
        # payloads are replaced by archive markers, while query-conditioned
        # retrieval restores only the evidence needed on this turn.
        if enable_history_replacement:
            # Remove old payloads from the model-facing cache. Their raw
            # contents remain recoverable from graph nodes or child archives.
            compact_history = mode == "legacy" or len(agent['main'].context()) > 0.75 * (
                agent['main'].prompt_ids_len + config.response_length
            )
            while compact_history and len(working_memory_turns) >= working_memory_keep_recent:
                old_turn = working_memory_turns.pop(0)
                if old_turn < len(agent['main'].chat):
                    agent['main'].replace_user_turn(
                        old_turn,
                        "[Archived working-memory payload; recoverable through ContextGraph.]",
                    )

            excluded = (
                {new_evidence_node_id} if new_evidence_node_id is not None else set()
            )
            retrieved = graph.retrieve_context(
                query=response,
                summary_budget=retrieval_summary_budget,
                evidence_budget=retrieval_evidence_budget,
                max_summaries=retrieval_max_summaries,
                max_evidence=retrieval_max_evidence,
                exclude_node_ids=excluded,
            )
            if retrieved:
                observation = (
                    f"{observation}\n\n"
                    "[ContextGraph retrieved working memory]\n"
                    f"{retrieved}"
                )
            retrieval_totals['calls'] += 1
            retrieval_totals['summary_tokens'] += graph.last_retrieval_stats.get(
                'summary_tokens', 0
            )
            retrieval_totals['evidence_tokens'] += graph.last_retrieval_stats.get(
                'evidence_tokens', 0
            )
            retrieval_totals['max_context_tokens'] = max(
                retrieval_totals['max_context_tokens'],
                graph.last_retrieval_stats.get('summary_tokens', 0)
                + graph.last_retrieval_stats.get('evidence_tokens', 0),
            )

        # The original protocol appends the complete graph after every action.
        # Long-horizon environments can disable this duplicate payload while
        # still receiving graph state after explicit controller checkpoints.
        if inject_graph_state_after_action:
            observation = (
                f"{observation}\n\n"
                "[Latest ContextGraph state]\n"
                f"{graph.to_state_text()}"
            )

        observation_turn = len(agent['main'].chat)
        fitted_observation = append_observation_preserving_final_answer(
            agent['main'],
            observation,
            final_answer_reserve,
            final_answer_safety_margin,
        )
        if fitted_observation is None:
            observation_budget_skips += 1
            pre_finalize_token_limit = bool(final_answer_reserve)
            break
        observation_budget_truncations += int(fitted_observation != str(observation))
        if enable_history_replacement:
            working_memory_turns.append(observation_turn)
        session_message.append({'role': 'user', 'content': fitted_observation})

        # Preserve the terminal observation, then stop before a same-turn
        # consolidation checkpoint can reject an already successful episode.
        if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
            natural_finish = True
            break

        # ── Forced consolidation checkpoint ──
        # Every `consolidation_interval` main turns, inject a checkpoint
        # asking the policy to emit a graph op or <pass>. <pass> is valid
        # (reward-neutral) only when the graph is saturated; otherwise
        # penalized. See ContextGraph.is_saturated() for thresholds.
        checkpoint_due = graph_checkpoint_due(
            main_turn_count,
            effective_max_turn,
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
                allow_pass = controller_allow_pass or graph.is_saturated()
                controller_prompt = graph_controller.action_prompt(
                    candidate_snapshot,
                    turn_id=main_turn_count,
                    allow_pass=allow_pass,
                    action_policy=controller_action_policy,
                )
                fitted_controller_prompt = append_observation_preserving_final_answer(
                    agent['main'],
                    controller_prompt,
                    final_answer_reserve,
                    final_answer_safety_margin,
                )
                if fitted_controller_prompt is None:
                    observation_budget_skips += 1
                    pre_finalize_token_limit = bool(final_answer_reserve)
                    break
                if not graph_controller.has_completion_budget(
                    remaining_generation_tokens(agent['main']),
                    protected_tokens=protected_final_answer_budget,
                ):
                    agent['main'].rollback(k=1)
                    consolidation_stats['budget_skips'] += 1
                    print(
                        '[GRAPH CONTROLLER SKIP] insufficient completion '
                        'token budget'
                    )
                    continue
                observation_budget_truncations += int(
                    fitted_controller_prompt != controller_prompt
                )
                session_message.append({
                    'role': 'user', 'content': fitted_controller_prompt,
                })

                controller_response = await step_preserving_final_answer(
                    agent['main'],
                    protected_final_answer_budget,
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
                    pre_finalize_token_limit = bool(
                        final_answer_reserve
                        and remaining_generation_tokens(agent['main'])
                        <= protected_final_answer_budget + 9
                    )
                    break
                session_message.append({
                    'role': 'assistant', 'content': controller_response,
                })
                controller_turn_idx = len(agent['main'].chat) - 1
                consolidation_stats['attempts'] += 1
                iteration += 1
                trace_before = graph_trace.capture(graph)
                decision_context = {
                    "mode": "controller_action",
                    "allow_pass": allow_pass,
                    **candidate_snapshot.trace_context(),
                }
                try:
                    decision_context["decision"] = json.loads(controller_response)
                except (TypeError, json.JSONDecodeError):
                    # Keep malformed output auditable. The controller will
                    # reject it below instead of allowing it to reach graph
                    # mutation code.
                    decision_context["decision_raw"] = controller_response
                graph_call = None
                repeat_select_hint = ""
                try:
                    graph_call = graph_controller.resolve_action(
                        graph,
                        candidate_snapshot,
                        controller_response,
                        allow_pass=allow_pass,
                        action_policy=controller_action_policy,
                    )
                    controller_action = graph_call['function']
                    repeat_select_hint = ""
                    if diagnostic_fix == "repeat" and controller_action == 'pass' and decision_context.get('decision', {}).get('action') == 'select':
                        repeat_select_hint = repeat_advice.observe('select_noop', {'node_id': graph.active_node_id})
                    if controller_action == 'pass':
                        controller_observation = GraphOpResult(
                            'Pass accepted: no graph edit was required.', True
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
                    graph_trace.record(
                        graph,
                        trace_before,
                        turn_id=main_turn_count,
                        source="model",
                        op=(
                            graph_call.get('function', 'unknown')
                            if graph_call is not None else 'unknown'
                        ),
                        args={"structured_response": controller_response},
                        success=False,
                        error=str(exc),
                        assistant_content=controller_response,
                        assistant_turn_index=controller_turn_idx,
                        decision_context=decision_context,
                    )
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
                    graph_trace.record(
                        graph,
                        trace_before,
                        turn_id=main_turn_count,
                        source="model",
                        op=controller_action,
                        args=graph_call['arguments'],
                        success=True,
                        assistant_content=controller_response,
                        assistant_turn_index=controller_turn_idx,
                        decision_context=decision_context,
                    )
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
                    "Your next response must use only an environment or branch "
                    "tool described in the system prompt, in its normal XML format."
                )
                if diagnostic_fix == "repeat" and graph_call is not None and repeat_select_hint:
                    controller_ack = repeat_select_hint + "\n\n" + controller_ack
                fitted_controller_ack = append_observation_preserving_final_answer(
                    agent['main'],
                    controller_ack,
                    final_answer_reserve,
                    final_answer_safety_margin,
                )
                if fitted_controller_ack is None:
                    observation_budget_skips += 1
                    pre_finalize_token_limit = bool(final_answer_reserve)
                    break
                observation_budget_truncations += int(
                    fitted_controller_ack != controller_ack
                )
                session_message.append({
                    'role': 'user', 'content': fitted_controller_ack,
                })

        if (
            checkpoint_due
            and checkpoint_budget_error is None
            and not structured_graph_controller
        ):
            is_sat = graph.is_saturated()
            n_active = len(graph.active_nodes)
            n_edges = len(graph.active_edges)
            n_nodes = len(graph.nodes)
            eligible_ids = [
                node.id for node in graph.active_nodes
                if node.id != graph.root_id
            ]
            eligible_text = ", ".join(eligible_ids) if eligible_ids else "(none)"
            examples = []
            if len(eligible_ids) >= 2:
                first, second = eligible_ids[:2]
                examples.extend([
                    f"<function=merge><parameter=node_ids>{first},{second}</parameter>"
                    "<parameter=summary>Combined relevant evidence</parameter></function>",
                    f"<function=add_edge><parameter=source>{first}</parameter>"
                    f"<parameter=target>{second}</parameter>"
                    "<parameter=relation>semantic</parameter></function>",
                ])
            if eligible_ids:
                examples.extend([
                    f"<function=select><parameter=node_id>{eligible_ids[0]}</parameter></function>",
                    f"<function=prune><parameter=node_id>{eligible_ids[0]}</parameter></function>",
                ])
            examples.append("<function=pass></function>")
            example_text = "\n".join(f"  {example}" for example in examples)
            pass_clause = (
                "If no obvious merge/prune/edge improvement helps, emit "
                "exactly <function=pass></function>."
                if is_sat else
                "You MUST emit a real operation; <function=pass> is NOT valid here "
                "and will be penalized."
            )
            consol_prompt = (
                f"[CONSOLIDATION CHECKPOINT turn={main_turn_count}]\n"
                f"Current graph: {n_active} active nodes, {n_edges} edges, {n_nodes} total. "
                f"Eligible node IDs: {eligible_text}.\n"
                f"Choose ONE operation: merge / prune / add_edge / select / pass. "
                f"Use only eligible IDs and exactly the parameter names shown below.\n"
                f"For merge, both node_ids and a meaningful non-empty summary are mandatory.\n"
                f"Valid XML forms using current IDs:\n{example_text}\n"
                f"{pass_clause}\n"
                f"After this checkpoint you continue the task normally."
            )
            fitted_consol_prompt = append_observation_preserving_final_answer(
                agent['main'],
                consol_prompt,
                final_answer_reserve,
                final_answer_safety_margin,
            )
            if fitted_consol_prompt is None:
                observation_budget_skips += 1
                pre_finalize_token_limit = bool(final_answer_reserve)
                break
            observation_budget_truncations += int(
                fitted_consol_prompt != consol_prompt
            )
            session_message.append({'role': 'user', 'content': fitted_consol_prompt})

            consol_response = await step_preserving_final_answer(
                agent['main'], protected_final_answer_budget
            )
            if consol_response is None:
                pre_finalize_token_limit = bool(
                    final_answer_reserve
                    and remaining_generation_tokens(agent['main'])
                    <= protected_final_answer_budget + 9
                )
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
                trace_before = graph_trace.capture(graph)
                budget_error = graph.graph_op_budget_error()
                consol_obs = (
                    GraphOpResult(f"[Error] {budget_error}.\n\n{graph.to_state_text()}", False)
                    if budget_error else handler(graph, consol_fn)
                )
                graph.record_graph_op(consol_obs.success)
                graph_trace.record(
                    graph, trace_before, turn_id=main_turn_count,
                    source="model", op=consol_fn['function'],
                    args=consol_fn.get('arguments', {}), success=consol_obs.success,
                    error=None if consol_obs.success else str(consol_obs).split("\n", 1)[0],
                    assistant_content=consol_response,
                    assistant_turn_index=consol_turn_idx,
                )
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
                trace_before = graph_trace.capture(graph)
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
                graph_trace.record(
                    graph, trace_before, turn_id=main_turn_count,
                    source="model", op="pass", args={}, success=is_sat,
                    error=None if is_sat else "pass rejected: graph not saturated",
                    assistant_content=consol_response,
                    assistant_turn_index=consol_turn_idx,
                )
            else:
                consolidation_stats['invalid'] += 1
                if process_reward and is_train:
                    agent['main'].set_process_reward(
                        consol_turn_idx, consolidation_invalid_penalty)
                print('[CONSOL INVALID]', str(consol_response)[:120])
                ack = "[CONSOLIDATION ACK] invalid response, continuing."

            ack = f"{ack}\n\n[Latest ContextGraph state]\n{graph.to_state_text()}"
            fitted_ack = append_observation_preserving_final_answer(
                agent['main'],
                ack,
                final_answer_reserve,
                final_answer_safety_margin,
            )
            if fitted_ack is None:
                observation_budget_skips += 1
                pre_finalize_token_limit = bool(final_answer_reserve)
                break
            observation_budget_truncations += int(fitted_ack != ack)
            session_message.append({'role': 'user', 'content': fitted_ack})

    finalizer_attempted = bool(
        final_answer_reserve
        and not (getattr(env, 'is_finish', False) or getattr(env, 'finish', False))
        and (not must_branch or bool(branches))
    )
    forced_finish = False
    if finalizer_attempted:
        if (
            structured_memory is not None
            and structured_memory.facts
            and structured_memory.revision != last_gap_revision
        ):
            await refresh_structured_memory_gaps("final")
        final_memory_overlay = structured_memory_overlay(query_text, final=True)
        finalizer_message_start = len(agent['main'].messages())
        forced_finish = await submit_emergency_final_answer(
            agent['main'],
            env,
            final_answer_reserve,
            run_action,
            context_overlay=final_memory_overlay,
        )
        session_message.extend(agent['main'].messages()[finalizer_message_start:])
        if forced_finish:
            mask_rollout = False

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
    agent_trajectories = serialize_agent_trajectories(agent)
    env.stats['get_final_score'] = score[1]
    env.stats['traj_num'] = len(agent)
    main_response_tokens = max(len(agent['main'].context()) - init_len, 0)
    main_context_tokens = len(agent['main'].context())
    working_context_limit = config.prompt_length + config.response_length
    env.stats['main_len'] = min(main_response_tokens, config.response_length)
    env.stats['main_context_tokens'] = main_context_tokens
    env.stats['working_context_limit'] = working_context_limit
    # This is post-run telemetry over the complete archived trajectory, not a
    # model input. Suppress the tokenizer model_max_length warning while still
    # counting the full untruncated trace.
    env.stats['total_token'] = len(tokenizer(
        print_chat(user_prompt + session_message),
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )['input_ids'])
    env.stats['main_turn'] = len(agent['main'].messages())
    env.stats['is_branch'] = int(len(agent) > 1)
    env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
    env.stats['use_all_branch'] = int(len(branches) + 1 > max_session)
    env.stats['natural_finish'] = int(natural_finish)
    env.stats['finalizer_attempted'] = int(finalizer_attempted)
    env.stats['forced_finish'] = int(forced_finish)
    env.stats['pre_finalize_token_limit'] = int(pre_finalize_token_limit)
    env.stats['observation_budget_truncations'] = observation_budget_truncations
    env.stats['observation_budget_skips'] = observation_budget_skips
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
    # Preserve the complete graph-reward decomposition for experiment
    # telemetry. These fields are diagnostics under GraphRPO; the optimized
    # reward remains selected below according to the configured estimator.
    graph_reward_stat_keys = {
        'compactness': 'graph_compactness',
        'structural': 'graph_structural',
        'merge_bonus': 'graph_merge_bonus',
        'prune_bonus': 'graph_prune_bonus',
        'usage_bonus': 'graph_usage_bonus',
        'uniqueness_bonus': 'graph_uniqueness_bonus',
        'uniqueness_raw': 'graph_uniqueness_raw',
        'cost_penalty': 'graph_cost_penalty',
        'invalid_op_penalty': 'graph_invalid_op_penalty',
        'bloat_penalty': 'graph_bloat_penalty',
        'operation_cost': 'graph_operation_cost',
        'total_ops': 'graph_total_ops',
        'n_folded': 'graph_n_folded',
        'n_pruned': 'graph_n_pruned',
        'n_cross_edges': 'graph_n_cross_edges',
    }
    for reward_key, stat_key in graph_reward_stat_keys.items():
        env.stats[stat_key] = float(graph_rewards.get(reward_key, 0.0))
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
    env.stats['controller_allow_pass'] = int(controller_allow_pass)
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
    # Isolated-variant specific stats: aggregate child subgraph sizes
    env.stats['isolated_n_subgraphs'] = len(branch_subgraph_stats)
    env.stats['isolated_total_subgraph_nodes'] = sum(s.get('n_total', 0) for s in branch_subgraph_stats.values())
    env.stats['isolated_total_subgraph_obs'] = sum(s.get('n_observations', 0) for s in branch_subgraph_stats.values())
    env.stats['memory_n_archives'] = len(graph.archives)
    env.stats['memory_history_replacement'] = float(enable_history_replacement)
    env.stats['memory_graph_state_after_action'] = float(
        inject_graph_state_after_action
    )
    env.stats['memory_archived_evidence'] = sum(
        len(archive.evidence) for archive in graph.archives.values()
    )
    env.stats['memory_retrieval_calls'] = retrieval_totals['calls']
    env.stats['memory_retrieval_summary_tokens'] = retrieval_totals['summary_tokens']
    env.stats['memory_retrieval_evidence_tokens'] = retrieval_totals['evidence_tokens']
    env.stats['memory_retrieval_max_context_tokens'] = retrieval_totals['max_context_tokens']
    env.stats['structured_memory_enabled'] = int(structured_memory_enabled)
    for stat_name, stat_value in structured_memory_stats.items():
        env.stats[f'structured_memory_{stat_name}'] = float(stat_value)
    memory_state = structured_memory.to_dict() if structured_memory is not None else None
    if structured_memory is not None:
        memory_summary_stats = structured_memory.stats()
        for stat_name in (
            'facts', 'links', 'nodes', 'structural_edges', 'goals', 'actions',
            'observations', 'gaps', 'can_answer', 'revision',
        ):
            env.stats[f'structured_memory_{stat_name}'] = float(
                memory_summary_stats[stat_name]
            )
        print(
            "[STRUCTURED MEMORY SUMMARY] "
            f"plans={structured_memory_stats['plan_calls']} "
            f"extractions={structured_memory_stats['extraction_calls']} "
            f"extraction_errors={structured_memory_stats['extraction_errors']} "
            f"facts={memory_summary_stats['facts']} "
            f"links={memory_summary_stats['links']} "
            f"gaps={memory_summary_stats['gaps']} "
            f"context_injections={structured_memory_stats['context_injections']} "
            f"max_snapshot_tokens="
            f"{structured_memory_stats['max_context_snapshot_tokens']}"
        )
    graph_trace_payload = graph_trace.finalize(graph)
    env.stats['graph_trace_events'] = len(graph_trace_payload['events'])
    env.stats['graph_trace_model_events'] = sum(
        event.get('source') == 'model'
        for event in graph_trace_payload['events']
    )
    tool_format_repair_log = [
        {"agent": agent_name, **repair}
        for agent_name, agent_instance in agent.items()
        for repair in agent_instance.tool_format_repairs
    ]
    env.stats['tool_format_repairs'] = len(tool_format_repair_log)

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
        max_turn=effective_max_turn,
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

        # Keep FoldAgent's paper process rewards as the shared base signal for
        # ContextGraph. Graph-specific outcome shaping is applied separately.
        GRAPH_OP_MARKERS = ('<function=merge>', '<function=prune>', '<function=select>', '<function=add_edge>')
        if rollout_status['unfolded_main']:
            bad_turn = [i for i, turn in enumerate(agent['main'].messages()) if
                        '<function=branch>' not in str(turn)
                        and not any(m in str(turn) for m in GRAPH_OP_MARKERS)]
            agent['main'].set_process_reward(bad_turn, -1)

        if rollout_status['no_finish']:
            last_completion = next(
                (
                    i for i in range(len(agent['main'].chat_completions) - 1, 0, -1)
                    if agent['main'].chat_completions[i] is not None
                ),
                None,
            )
            if last_completion is not None:
                agent['main'].set_process_reward(last_completion, -1)

        # In the isolated variant, trajectories that skip graph ops are not
        # penalized. Graph reward is outcome-only via compute_graph_reward().
        if 'scope' in process_reward:
            env.stats['scope_judge'] = 1
            for name in branches:
                assigned_task = branch_tasks[name]
                return_message = branch_return.get(name, '')
                is_focus, justification = await judge_scope(assigned_task, return_message)
                if is_focus < 0:
                    print(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                    agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                    env.stats['scope_judge'] = 0

        ERR_MARKERS = (
            'Failed to validate tool call',
            'Failed to parse tool call',
            'You are in branch mode and cannot branch task or finish the task.',
            'No function call was detected in the model response',
            '[Error] The "search" function requires a "query" argument',
            '[Error] The "open_page" function requires either a "docid" or a "url".',
            '[Error] The function',
        )
        for name in agent:
            for i, turn in enumerate(agent[name].chat):
                if any(m in str(turn) for m in ERR_MARKERS):
                    agent[name].set_process_reward(max(i - 1, 0), -1)

        if score[1] <= 0:
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

            if 'reward_scope' in process_reward and 'scope' not in process_reward:
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

    reference_edit_requests = []
    if graph_rpo_enabled and is_train:
        if graph_rpo_backend == EXTERNAL_EVALUATOR_BACKEND:
            graph_rpo_metrics = await assign_graph_edit_credits(
                agent=agent['main'],
                graph_trace=graph_trace_payload,
                question=query_text,
                terminal_reward=score[1],
                tokenizer=tokenizer,
                plugin_config=config.plugin,
            )
        elif graph_rpo_backend == OLD_POLICY_COUNTERFACTUAL_QA_BACKEND:
            if not hasattr(env, 'score_answer'):
                raise GraphRPOEvaluatorError(
                    "old_policy_counterfactual_qa requires an environment "
                    "with score_answer(predicted_answer, audit_sink)"
                )

            counterfactual_max_tokens = int(
                getattr(config.plugin, 'graph_rpo_counterfactual_max_new_tokens', 512)
            )
            counterfactual_temperature = float(
                getattr(config.plugin, 'graph_rpo_counterfactual_temperature', 1.0)
            )
            counterfactual_top_p = float(
                getattr(config.plugin, 'graph_rpo_counterfactual_top_p', 1.0)
            )
            raw_counterfactual_thinking = getattr(
                config.plugin, 'graph_rpo_counterfactual_enable_thinking', False
            )
            if isinstance(raw_counterfactual_thinking, str):
                counterfactual_enable_thinking = (
                    raw_counterfactual_thinking.strip().lower()
                    in ('1', 'true', 'yes', 'on')
                )
            else:
                counterfactual_enable_thinking = bool(raw_counterfactual_thinking)

            async def generate_counterfactual_answer(graph_view, sample_index, seed):
                probe_messages = format_counterfactual_qa_messages(
                    query_text, graph_view
                )
                probe_agent = Agent(
                    llm_client,
                    probe_messages,
                    tokenizer,
                    config,
                    prompt_turn=len(probe_messages),
                    process_reward_min_precedence=True,
                    chat_template_kwargs={
                        'enable_thinking': counterfactual_enable_thinking,
                    },
                )
                return await probe_agent.step(
                    max_new_tokens=counterfactual_max_tokens,
                    completion_kwargs={
                        'sampling_params': {
                            'temperature': counterfactual_temperature,
                            'top_p': counterfactual_top_p,
                            'max_tokens': counterfactual_max_tokens,
                            'seed': seed,
                        }
                    },
                ) or ''

            async def score_counterfactual_answer(answer, audit_sink):
                return await env.score_answer(answer, audit_sink=audit_sink)

            graph_rpo_metrics = await assign_counterfactual_graph_edit_credits(
                agent=agent['main'],
                graph_trace=graph_trace_payload,
                question=query_text,
                generate_answer=generate_counterfactual_answer,
                score_answer=score_counterfactual_answer,
                plugin_config=config.plugin,
            )
        elif graph_rpo_backend in ANSWER_LIKELIHOOD_BACKENDS:
            reference_edit_requests, graph_rpo_metrics = prepare_reference_graph_edit_requests(
                graph_trace=graph_trace_payload,
                terminal_reward=score[1],
                credit_backend=graph_rpo_backend,
            )
        else:  # pragma: no cover - graph_rpo_credit_backend validates this.
            raise ValueError(f"unsupported GraphRPO credit backend: {graph_rpo_backend}")
        env.stats.update(graph_rpo_metrics)

    use_graph_reward = (
        not graph_rpo_enabled and process_reward and 'graph' in process_reward
    )

    for name in agent if is_train else ['main']:
        out = await agent[name].get_data()
        reference_edits_with_spans = []
        if (
            graph_rpo_enabled
            and graph_rpo_backend in ANSWER_LIKELIHOOD_BACKENDS
            and name == 'main'
        ):
            turn_token_indices = out['response_turn_token_indices']
            for request in reference_edit_requests:
                token_indices = turn_token_indices.get(request['assistant_turn_index'], [])
                if token_indices:
                    reference_edits_with_spans.append({
                        **request,
                        'response_token_indices': token_indices,
                    })
            env.stats['graph_rpo_creditable_edits'] = len(reference_edits_with_spans)
        agent_reward = score[1]

        # Every trajectory emitted from one gen_uid is another view of the
        # same episode. Give main and branches the same terminal graph reward;
        # branch-specific quality remains represented by token process rewards.
        if use_graph_reward:
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
                'judge_audit': copy.deepcopy(getattr(env, 'judge_audit', [])),
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
                'extra_info': copy.deepcopy(_get(item.non_tensor_batch['extra_info'])),
                'model_contexts': agent[name].model_contexts,
                'branch_model_contexts': {key: value.model_contexts for key, value in agent.items() if key != 'main'} if name == 'main' else {},
                'contextgraph_memory_mode': mode,
                'diagnostic_fix': diagnostic_fix,
                'repeat_advice_count': repeat_advice.warnings,
                'message_str': print_chat(session_message),
                'meta_info': f"N: {len(agent)} | {name} | G:{len(graph.nodes)}n/{len(graph.active_edges)}e [iso]",
                'process_reward_mask': out['process_reward_mask'],
                **(
                    {'graph_edit_credit_mask': out['graph_edit_credit_mask']}
                    if graph_rpo_enabled
                    else {}
                ),
                **(
                    {
                        'graph_rpo_reference_edits': reference_edits_with_spans,
                        'graph_rpo_reference_question': query_text,
                        'graph_rpo_reference_answer': getattr(env, 'label_answer', None),
                    }
                    if graph_rpo_enabled
                    and graph_rpo_backend in ANSWER_LIKELIHOOD_BACKENDS
                    else {}
                ),
                'uid': uid,
                'gen_uid': gen_uid,
                'graph_state': graph.to_state_text(),
                'structured_memory': copy.deepcopy(memory_state),
                'graph_trace': graph_trace_payload,
                'graph_rewards': graph_rewards,
                'tool_format_repairs': copy.deepcopy(tool_format_repair_log),
                'isolated_subgraph_stats': branch_subgraph_stats,
                'agent_trajectories': copy.deepcopy(agent_trajectories),
            }
        )
        outs.append(copy.deepcopy(out))

    if max_traj is not None and len(outs) > max_traj:
        idx = [0] + sorted(random.sample(range(1, len(outs)), k=max_traj - 1))
        outs = [outs[i] for i in idx]

    return outs
