"""SUPO Algorithm 2 inference adaptation; no SUPO RL update or trained weights.

Reference: https://arxiv.org/abs/2510.06727v1, Algorithm 2. A threshold-crossing
action/observation is omitted from working context, but its tool side effects
are NOT rolled back. This adapter is consequently limited to local search.
"""
from __future__ import annotations

import asyncio
import copy
import re
import time

from .agentfold_agent import _fit_observation, _get, parse_response
from .environment_lifecycle import managed_environment
from .prompts import create_chat
from .utils import AgentLoopMetrics, AgentLoopOutput, _apply_chat_template, select_env


PROVENANCE = {
    "method": "supo", "variant": "zero_shot_rollout_adaptation_v1",
    "paper": "https://arxiv.org/abs/2510.06727v1", "algorithm": 2,
    "implementation": "paper_based_reimplementation",
    "supo_rl_training": False, "trained_supo_checkpoint": False,
    "budget": "cumulative generated IDs plus visible observation/instruction text; no refunds",
    "overflow": "discard crossing action/observation from working context only",
}
SUMMARY_REQUEST = (
    "Summarize the previous research so that you can continue in a fresh context. "
    "Preserve verified evidence and source docids, unresolved constraints, failed "
    "searches, and useful next steps. Do not invent findings or call tools. "
    "Return only <summary>your task-relevant notes</summary>."
)


def settings(plugin):
    values = {"supo_context_threshold": 16384, "supo_max_summaries": 2,
              "supo_summary_max_tokens": 1024}
    for key, default in values.items():
        value = getattr(plugin, key, default)
        if type(value) is not int or value < (0 if key == "supo_max_summaries" else 1):
            raise ValueError(f"Invalid {key}: {value}")
        values[key] = value
    return values


def parse_summary(text):
    visible = text.rsplit("</think>", 1)[-1].strip()
    match = re.fullmatch(r"<summary>(.*?)</summary>", visible, re.S)
    if (not match or not match.group(1).strip()
            or visible.count("<summary>") != 1 or visible.count("</summary>") != 1):
        raise ValueError("Expected one nonempty <summary> block")
    return match.group(1).strip()


async def process_item(item, context):
    if context.is_train:
        raise ValueError("SUPO rollout adaptation is evaluation-only; joint RL is not implemented")
    ability = _get(item.non_tensor_batch["ability"])
    if ability not in {"LocalSearch", "GAIA"}:
        raise ValueError("SUPO adaptation supports only local search / text-only GAIA")
    config = copy.deepcopy(context.config.actor_rollout_ref.rollout)
    plugin, tokenizer = config.plugin, context.tokenizer
    options = settings(plugin)
    threshold = options["supo_context_threshold"]
    max_summaries = options["supo_max_summaries"]
    summary_cap = options["supo_summary_max_tokens"]
    budget = int(getattr(plugin, "val_response_length", None) or config.response_length)
    window = int(config.prompt_length) + budget
    max_turn = int(getattr(plugin, "val_max_turn", plugin.max_turn))
    turn_cap = int(getattr(plugin, "turn_max_new_tokens", 2048))
    reserve = max(0, int(getattr(plugin, "final_answer_reserve", 0)))
    margin = max(0, int(getattr(plugin, "final_answer_safety_margin", 64)))
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    render = lambda messages: list(_apply_chat_template(tokenizer, messages, config,
                                     tokenize=True, add_generation_prompt=True))
    # A summary request must fit even when the retained context is just below L.
    instruction_cost = len(render([{"role": "user", "content": SUMMARY_REQUEST}]))
    if threshold + instruction_cost + summary_cap > window:
        raise ValueError("SUPO threshold leaves insufficient working context for summary generation")
    started = time.monotonic()
    deadline = started + float(getattr(plugin, "session_timeout", 3600))
    env = select_env(ability, config)(config, tokenizer, ability)
    env.raise_judge_errors = True
    async with managed_environment(env):
        await env.init_env(item)
        fixed = create_chat(env.instance_info["problem_statement"], "search", item)
        if len(render(fixed)) >= threshold:
            raise ValueError("SUPO threshold must exceed the full initial prompt; task is never truncated")
        working, transcript = copy.deepcopy(fixed), copy.deepcopy(fixed)
        audit = []
        stats = dict(environment_steps=0, summary_attempts=0, summary_restarts=0,
                     invalid_tool=0, invalid_summary=0, generated_tokens=0,
                     observation_tokens=0, instruction_tokens=0, supo_discarded_rounds=0,
                     supo_discarded_observation_tokens=0, observation_budget_truncations=0,
                     observation_budget_skips=0)
        pending, feedback, used, iteration = False, "", 0, 0
        reason = "max_turn"
        input_ids, output_ids, logprobs = [], [], []
        while iteration < max_turn:
            remaining = budget - used
            finish_only = bool(reserve and (remaining <= reserve + margin + max(turn_cap, 10)
                                           or iteration == max_turn - 1))
            phase = "final" if finish_only else ("summary" if pending else "action")
            messages = copy.deepcopy(working)
            if phase == "summary":
                instruction = SUMMARY_REQUEST
            elif phase == "final":
                instruction = "Budget ending: submit one finish call now using the available evidence."
            else:
                instruction = feedback
            if instruction:
                instruction = _fit_observation(instruction, tokenizer, max(0, budget - used))
                messages.append({"role": "user", "content": instruction})
                cost = len(encode(instruction))
                used += cost
                stats["instruction_tokens"] += cost
            candidate_ids = render(messages)
            limit = min(window - len(candidate_ids), budget - used
                        - (reserve + margin if reserve and not finish_only else 0))
            if turn_cap > 0:
                limit = min(limit, turn_cap)
            if phase == "summary":
                limit = min(limit, summary_cap)
            if limit < 10:
                reason = "token_limit"
                break
            if deadline <= time.monotonic():
                reason = "timeout"
                break
            iteration += 1
            try:
                result = await asyncio.wait_for(context.llm_client.create_completion(
                    candidate_ids, messages=messages, max_new_tokens=limit,
                    max_len=len(candidate_ids) + limit), timeout=deadline - time.monotonic())
                if result is None:
                    reason = "token_limit"
                    break
                response = result["choices"][0]["message"]
                input_ids, output_ids = candidate_ids, list(response["raw_output_ids"])
                logprobs = list(response.get("response_log_probs") or [0.] * len(output_ids))
                if not output_ids or len(output_ids) > limit or len(logprobs) != len(output_ids):
                    raise RuntimeError("model completion violated token budget/alignment")
                used += len(output_ids)
                stats["generated_tokens"] += len(output_ids)
                text = response["content"]
                record = dict(phase=phase, input_ids=input_ids, output_ids=output_ids,
                              messages=messages, response=text, max_tokens=limit)
                audit.append(record)
                if instruction:
                    transcript.append({"role": "user", "content": instruction})
                transcript.append({"role": "assistant", "content": text})
                if phase == "summary":
                    stats["summary_attempts"] += 1
                    try:
                        summary = parse_summary(text)
                        resumed = copy.deepcopy(fixed)
                        resumed[-1]["content"] += "\n\nPrevious research summary:\n" + summary
                        if len(render(resumed)) >= threshold:
                            raise ValueError("Summary does not reduce context below the threshold")
                    except ValueError as exc:
                        record["format_error"] = str(exc)
                        stats["invalid_summary"] += 1
                        reason = "invalid_summary"
                        break
                    working, pending, feedback = resumed, False, ""
                    stats["summary_restarts"] += 1
                    record["resumed_context_tokens"] = len(render(working))
                    continue
                try:
                    # Shared XML validation only; AgentFold compression is not enabled.
                    action, _ = parse_response(text, [], finish_only=finish_only)
                except (ValueError, TypeError) as exc:
                    stats["invalid_tool"] += 1
                    feedback = record["format_error"] = str(exc)
                    continue
                feedback = ""
                record["action"] = action
                result = await asyncio.wait_for(env.run_action(action),
                    timeout=max(0, deadline - time.monotonic()))
                stats["environment_steps"] += 1
                if getattr(env, "env_fail", False):
                    raise RuntimeError("SUPO environment failed")
                if getattr(env, "is_finish", False):
                    reason = "finish"
                    break
                raw = str(result.get("observation", "Empty"))
                record["raw_observation"] = raw
                transcript.append({"role": "user", "content": raw})
                candidate = working + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": raw}]
                record["candidate_context_tokens"] = len(render(candidate))
                if record["candidate_context_tokens"] >= threshold:
                    # Algorithm 2 line 11: retain s_t, not (s_t, a_t, o_t).
                    # The real tool has already run and is never replayed/undone.
                    record["discarded_from_working_context"] = True
                    stats["supo_discarded_rounds"] += 1
                    stats["supo_discarded_observation_tokens"] += len(encode(raw))
                    if stats["summary_restarts"] >= max_summaries:
                        reason = "summary_limit"
                        break
                    pending = True
                    continue
                shown = _fit_observation(raw, tokenizer, max(0, budget - used - reserve - margin))
                cost = len(encode(shown))
                used += cost
                stats["observation_tokens"] += cost
                stats["observation_budget_truncations"] += int(bool(shown) and shown != raw)
                stats["observation_budget_skips"] += int(not shown)
                candidate[-1]["content"] = shown
                record["shown_observation"] = shown
                working = candidate
            except asyncio.TimeoutError:
                reason = "timeout"
                break
        _, reward, _ = await asyncio.wait_for(env.get_reward(item, transcript, context), timeout=600)
        stats.update(dict(getattr(env, "stats", {})))
        stats.update(task_reward=float(reward), main_turn=iteration, main_len=used,
                     main_context_tokens=len(render(working)), working_context_limit=window,
                     session_time=time.monotonic() - started, model_requests=len(audit),
                     total_token=sum(len(r["input_ids"]) + len(r["output_ids"]) for r in audit),
                     hit_token_limit=int(reason == "token_limit"), hit_max_turn=int(reason == "max_turn"),
                     hit_summary_limit=int(reason == "summary_limit"), hit_timeout=int(reason == "timeout"))
        return AgentLoopOutput(prompt_ids=input_ids, response_ids=output_ids,
            response_mask=[0] * len(output_ids), response_logprobs=logprobs,
            reward_score=float(reward), num_turns=iteration, metrics=AgentLoopMetrics(),
            extra_fields={"messages": transcript, "model_contexts": audit, "env_stats": stats,
                          "is_finish": bool(getattr(env, "is_finish", False)), "termination_reason": reason,
                          "judge_audit": copy.deepcopy(getattr(env, "judge_audit", [])),
                          "supo": {**PROVENANCE, **options}, "working_history": working})
