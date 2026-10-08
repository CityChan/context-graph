"""Shared, audited BC-P runner for the MemoBrain / A-MEM zero-shot adapters."""
from __future__ import annotations

import asyncio
import copy
import json
import time

from . import graph_memory_baselines as memory
from .agentfold_agent import _fit_observation, _get, parse_response
from .environment_lifecycle import managed_environment
from .prompts import create_chat
from .single_tool_protocol import MAX_FORMAT_ERRORS, PROTOCOL, retry_context
from .utils import AgentLoopMetrics, AgentLoopOutput, _apply_chat_template, select_env


class BudgetEnd(Exception):
    pass


def provenance(method, options):
    return {**memory.PROVENANCE[method], **options, **PROTOCOL, "method": method,
            "variant": "zero_shot_bcp_adaptation_v2", "trained_checkpoint": False,
            "memory_scope": "one task; no cross-task state", "helper_model": "same actor endpoint",
            "budget": "all actor/helper generated tokens + visible observations + control instructions; no refunds",
            "turns": "all actor and memory model requests; final answer slot reserved"}


async def process_item(item, context):
    if context.is_train:
        raise ValueError("Graph memory baselines are evaluation-only")
    ability = _get(item.non_tensor_batch["ability"])
    if ability not in {"LocalSearch", "GAIA"}:
        raise ValueError("Graph memory baselines require LocalSearch or text-only GAIA")
    config = copy.deepcopy(context.config.actor_rollout_ref.rollout)
    plugin, tokenizer = config.plugin, context.tokenizer
    method = plugin.workflow.removeprefix("search_")
    if method not in {"memobrain", "amem"}:
        raise ValueError(f"Unknown memory baseline: {method}")
    options = memory.settings(plugin)
    budget = int(getattr(plugin, "val_response_length", None) or config.response_length)
    window = int(config.prompt_length) + int(config.response_length)
    max_turn = int(getattr(plugin, "val_max_turn", plugin.max_turn))
    turn_cap = int(getattr(plugin, "turn_max_new_tokens", 2048))
    reserve = max(0, int(getattr(plugin, "final_answer_reserve", 0)))
    margin = max(0, int(getattr(plugin, "final_answer_safety_margin", 64)))
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    render = lambda m: list(_apply_chat_template(tokenizer, m, config, tokenize=True, add_generation_prompt=True))
    started = time.monotonic()
    deadline = started + float(getattr(plugin, "session_timeout", 3600))
    stats = dict(environment_steps=0, invalid_tool=0, invalid_memory=0, memory_requests=0,
                 memory_updates=0, memory_recalls=0, generated_tokens=0, observation_tokens=0,
                 instruction_tokens=0, observation_budget_truncations=0, observation_budget_skips=0,
                 memory_retrieval_calls=0, memory_notes_omitted=0)
    audit, memory_audit = [], []
    used, iteration = 0, 0
    last_ids, last_outputs, last_logprobs = [], [], []

    async def request(messages, phase, final=False, instruction=""):
        nonlocal used, iteration, last_ids, last_outputs, last_logprobs
        if iteration >= max_turn or (phase.startswith("memory") and iteration >= max_turn - 1):
            raise BudgetEnd("max_turn")
        candidate = render(messages)
        instruction_cost = len(encode(instruction))
        limit = min(window - len(candidate), budget - used - instruction_cost
                    - (reserve + margin if not final else 0))
        if turn_cap > 0:
            limit = min(limit, turn_cap)
        if phase.startswith("memory"):
            limit = min(limit, options["memory_helper_max_tokens"])
        if limit < 10:
            raise BudgetEnd("token_limit")
        iteration += 1
        result = await asyncio.wait_for(context.llm_client.create_completion(
            candidate, messages=messages, max_new_tokens=limit, max_len=len(candidate) + limit),
            timeout=max(0, deadline - time.monotonic()))
        if result is None:
            raise BudgetEnd("token_limit")
        response = result["choices"][0]["message"]
        outputs = list(response["raw_output_ids"])
        logprobs = list(response.get("response_log_probs") or [0.] * len(outputs))
        if not outputs or len(outputs) > limit or len(outputs) != len(logprobs):
            raise RuntimeError("Model completion violated budget/alignment")
        used += len(outputs) + instruction_cost
        stats["generated_tokens"] += len(outputs)
        stats["instruction_tokens"] += instruction_cost
        stats["memory_requests"] += int(phase.startswith("memory"))
        last_ids, last_outputs, last_logprobs = candidate, outputs, logprobs
        audit.append(dict(phase=phase, input_ids=candidate, output_ids=outputs,
                          messages=copy.deepcopy(messages), response=response["content"], max_tokens=limit))
        return response["content"]

    async def helper(prompt, data, phase, apply):
        text = await request([{"role": "system", "content": prompt},
                              {"role": "user", "content": json.dumps(data, ensure_ascii=False)}],
                             phase, instruction=prompt)
        record = dict(phase=phase, request_index=len(audit) - 1, applied=False)
        memory_audit.append(record)
        try:
            decision = memory.json_object(text)
            apply(decision)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            record["error"] = str(exc)
            stats["invalid_memory"] += 1
            return False
        record.update(applied=True, decision=decision)
        stats["memory_updates"] += 1
        return True

    env = select_env(ability, config)(config, tokenizer, ability)
    env.raise_judge_errors = True
    async with managed_environment(env):
        await env.init_env(item)
        task = env.instance_info["problem_statement"]
        fixed = create_chat(task, "search_single", item)
        if len(render(fixed)) + max(reserve, 10) >= window:
            raise ValueError("Original task does not fit; never truncate it")
        store = memory.MemoBrainMemory(task) if method == "memobrain" else memory.AMemMemory()
        transcript, episodes, working = copy.deepcopy(fixed), [], copy.deepcopy(fixed)
        feedback, reason, force_final = "", "max_turn", False
        rejected, consecutive_invalid = "", 0
        try:
            while iteration < max_turn:
                finish_only = force_final or iteration == max_turn - 1 or budget - used <= reserve + margin + max(turn_cap, 10)
                if method == "memobrain":
                    working = copy.deepcopy(fixed) + store.history()
                else:
                    recent = [m for pair in episodes[-2:] for m in pair]
                    # Task plus last action asks about the current subproblem, without gold labels.
                    query = task + ("\n" + episodes[-1][0]["content"] if episodes else "")
                    recalled = store.retrieve(query, options["amem_topk"])
                    stats["memory_retrieval_calls"] += 1
                    selected, room = [], options["amem_memory_tokens"]
                    for note in recalled:
                        visible = {k: v for k, v in note.items() if k != "evolution_history"}
                        text = json.dumps(visible, ensure_ascii=False)
                        if len(encode(text)) <= room:
                            selected.append(text)
                            room -= len(encode(text))
                        else:
                            stats["memory_notes_omitted"] += 1
                    working = copy.deepcopy(fixed)
                    if selected:
                        working.append({"role": "user", "content": "Retrieved A-MEM notes (observed evidence):\n" + "\n".join(selected)})
                    working.extend(recent)
                instruction = ("Budget ending: submit one finish call now using available evidence."
                               if finish_only else feedback)
                if rejected:
                    working.append({"role": "assistant", "content": rejected})
                if instruction:
                    working.append({"role": "user", "content": instruction})
                text = await request(working, "final" if finish_only else "action", finish_only, instruction)
                transcript.append({"role": "assistant", "content": text})
                try:
                    action, _ = parse_response(text, [], finish_only=finish_only)
                except (ValueError, TypeError) as exc:
                    stats["invalid_tool"] += 1
                    audit[-1]["format_error"] = str(exc)
                    consecutive_invalid += 1
                    if consecutive_invalid >= MAX_FORMAT_ERRORS:
                        reason = "invalid_tool_limit"
                        break
                    rejected, feedback = retry_context(text, str(exc), tokenizer)
                    continue
                feedback, rejected, consecutive_invalid = "", "", 0
                audit[-1]["action"] = action
                result = await asyncio.wait_for(env.run_action(action), timeout=max(0, deadline - time.monotonic()))
                stats["environment_steps"] += 1
                if getattr(env, "env_fail", False):
                    raise RuntimeError("Graph memory baseline environment failed")
                if getattr(env, "is_finish", False):
                    reason = "finish"
                    break
                raw = str(result.get("observation", "Empty"))
                shown = _fit_observation(raw, tokenizer, max(0, budget - used - reserve - margin))
                cost = len(encode(shown))
                used += cost
                stats["observation_tokens"] += cost
                stats["observation_budget_truncations"] += int(bool(shown) and shown != raw)
                stats["observation_budget_skips"] += int(not shown)
                audit[-1].update(raw_observation=raw, shown_observation=shown)
                pair = [{"role": "assistant", "content": text}, {"role": "user", "content": shown}]
                episodes.append(pair)
                transcript.append(pair[1])
                if method == "memobrain":
                    store.append(pair)
                # Reserve a final actor request even when maintenance exhausts the budget.
                try:
                    if method == "memobrain":
                        await helper(memory.MEMORIZE, dict(interaction=pair, graph=store.state()),
                                     "memory_memorize", store.patch)
                        if (len(episodes) % options["memobrain_recall_interval"] == 0
                                or len(render(fixed + store.history())) >= options["memobrain_context_threshold"]):
                            stats["memory_recalls"] += 1
                            await helper(memory.RECALL, store.state(), "memory_recall", store.maintain)
                    else:
                        content = json.dumps(pair, ensure_ascii=False)
                        neighbors = store.nearest(content, options["amem_topk"])
                        ok = await helper(memory.ANALYZE, dict(interaction=pair), "memory_analyze",
                                          lambda analysis: store.add(content, analysis))
                        if not ok:
                            # Keep observed content retrievable even if metadata JSON failed.
                            store.add(content, dict(keywords=[], context="Unparsed metadata; raw evidence retained", tags=[]))
                            memory_audit[-1]["fallback"] = "raw_note_without_metadata"
                        if ok and neighbors:
                            nid = next(reversed(store.nodes))
                            await helper(memory.EVOLVE, dict(new_note=store.nodes[nid],
                                neighbors=[store.nodes[n] for n in neighbors]), "memory_evolve",
                                lambda decision: store.evolve(nid, neighbors, decision))
                except BudgetEnd:
                    force_final = True
        except BudgetEnd as exc:
            reason = str(exc)
        except asyncio.TimeoutError:
            reason = "timeout"
        _, reward, _ = await asyncio.wait_for(env.get_reward(item, transcript, context), timeout=600)
        stats.update(dict(getattr(env, "stats", {})))
        stats.update(task_reward=float(reward), main_turn=iteration, main_len=used,
                     main_context_tokens=len(render(working)), working_context_limit=window,
                     session_time=time.monotonic() - started, model_requests=len(audit),
                     total_token=sum(len(r["input_ids"]) + len(r["output_ids"]) for r in audit),
                     actor_requests=len(audit) - stats["memory_requests"],
                     embedding_calls=getattr(store, "embedding_calls", 0),
                     memory_nodes=len(store.nodes) - int(method == "memobrain"),
                     memory_edges=(len(store.edges) if method == "memobrain" else
                                   sum(len(n["links"]) for n in store.nodes.values())),
                     memobrain_folds=sum(len(r.get("decision", {}).get("fold_ops", [])) for r in memory_audit if r["applied"]),
                     memobrain_flushes=sum(len(r.get("decision", {}).get("flush_ops", [])) for r in memory_audit if r["applied"]),
                     amem_evolutions=sum(bool(r.get("decision", {}).get("should_evolve")) for r in memory_audit if r["applied"]),
                     hit_token_limit=int(reason == "token_limit"), hit_max_turn=int(reason == "max_turn"),
                     hit_timeout=int(reason == "timeout"))
        stats["hit_format_retry_limit"] = int(reason == "invalid_tool_limit")
        return AgentLoopOutput(prompt_ids=last_ids, response_ids=last_outputs,
            response_mask=[0] * len(last_outputs), response_logprobs=last_logprobs,
            reward_score=float(reward), num_turns=iteration, metrics=AgentLoopMetrics(),
            extra_fields={"messages": transcript, "model_contexts": audit, "env_stats": stats,
                          "is_finish": bool(getattr(env, "is_finish", False)), "termination_reason": reason,
                          "judge_audit": copy.deepcopy(getattr(env, "judge_audit", [])),
                          "baseline_protocol": provenance(method, options), "memory_state": store.state(),
                          "memory_audit": memory_audit, "working_history": working})
