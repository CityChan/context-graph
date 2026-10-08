"""Evaluation-only, AgentFold-style zero-shot adaptation (not FoldAgent).

Reimplements suffix folding from Alibaba-NLP/DeepResearch WebAgent/AgentFold,
revision f72f75d8c3eb842f2bbbab096a12206ff66e270f. Uses this repo's search tools,
backbone and judge, not the authors' trained policy or web-service stack.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import time
from dataclasses import dataclass

import numpy as np

from .environment_lifecycle import managed_environment
from .prompts import create_chat
from .single_tool_protocol import MAX_FORMAT_ERRORS, PROTOCOL, retry_context
from .utils import AgentLoopMetrics, AgentLoopOutput, _apply_chat_template, select_env


PROVENANCE = {
    "method": "agentfold", "variant": "zero_shot_adaptation_v2", **PROTOCOL,
    "upstream_revision": "f72f75d8c3eb842f2bbbab096a12206ff66e270f",
    "upstream_source": "Alibaba-NLP/DeepResearch/WebAgent/AgentFold/infer.py",
    "trained_agentfold_checkpoint": False,
    "budget": "cumulative generated IDs plus tokenized observations/feedback; folding never refunds tokens",
}

FOLD_PROMPT = """
You also manage your working history using AgentFold-style compression.
Before each search/open_page call after the first completed step, emit exactly:
<compress>{"compress_range":[START,END],"compress_text":"faithful summary"}</compress>
Then emit exactly one existing <function=...> call, with its parameters.
START and END are inclusive integer step IDs. Select a contiguous suffix ending
at the latest step, starting at the beginning of an existing block. Never split
a compressed block. Summarize only observed information, preserving source IDs,
evidence, failed searches and unresolved questions. A singleton range condenses
the latest step; a larger suffix can combine earlier summaries with new evidence.
The first call has no compress block. A finish call needs no compression.
You may reason before the compress block; do not put tool calls inside summaries.
The original question stays available. Compressed history replaces the selected
steps in future inputs; the discarded details cannot be retrieved from history.
""".strip()


@dataclass(frozen=True)
class Step:
    start: int
    end: int
    content: str
    compressed: bool = False

    def render(self):
        span = str(self.start) if self.start == self.end else f"{self.start} to {self.end}"
        return f"[{'Compressed Step' if self.compressed else 'Step'} {span}]\n{self.content}"


def fold_suffix(steps: list[Step], decision: dict) -> list[Step]:
    """Validate without mutating state; summaries may only replace whole blocks."""
    if not isinstance(decision, dict) or set(decision) != {"compress_range", "compress_text"}:
        raise ValueError("compress requires exactly compress_range and compress_text")
    span, summary = decision["compress_range"], decision["compress_text"]
    if (not isinstance(span, list) or len(span) != 2
            or any(type(v) is not int for v in span)):
        raise ValueError("compress_range must contain two integer step IDs")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("compress_text must be a nonempty string")
    if not steps or span[1] != steps[-1].end:
        raise ValueError("compression must end at the latest completed step")
    index = next((i for i, step in enumerate(steps) if step.start == span[0]), None)
    if index is None or any(a.end + 1 != b.start for a, b in zip(steps, steps[1:])):
        raise ValueError("compression must select contiguous whole blocks")
    return steps[:index] + [Step(span[0], span[1], summary.strip(), True)]


def parse_response(text: str, steps: list[Step], *, finish_only=False):
    """Do not dispatch tools until the entire response's fold/action is valid."""
    body = text.rsplit("</think>", 1)[-1].strip()
    folds = re.findall(r"<compress>(.*?)</compress>", body, re.S)
    if body.count("<compress>") != len(folds) or len(folds) > 1:
        raise ValueError("expected at most one complete compress block")
    action_body = re.sub(r"<compress>.*?</compress>", "", body, flags=re.S)
    calls = list(re.finditer(r"<function=([^>]+)>(.*?)</function>", action_body, re.S))
    if len(calls) != 1 or action_body.count("<function=") != 1:
        raise ValueError("emit exactly one complete function call")
    call = calls[0]
    name = call.group(1)
    params = re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", call.group(2), re.S)
    args = dict(params)
    allowed = {"search": {"query", "topk"}, "open_page": {"docid", "url"},
               "finish": {"answer", "explanation", "confidence"}}
    if name not in allowed or (finish_only and name != "finish"):
        raise ValueError("finish is required now" if finish_only else "unsupported function")
    if (len(args) != len(params) or set(args) - allowed[name]
            or re.sub(r"<parameter=[^>]+>.*?</parameter>", "", call.group(2), flags=re.S).strip()):
        raise ValueError("invalid or duplicate tool parameters")
    required = {"search": ("query",), "open_page": ("docid", "url"), "finish": ("answer",)}
    if not any(args.get(key, "").strip() for key in required[name]):
        raise ValueError("missing required tool argument")
    from envs.local_search import extract_fn_call
    if extract_fn_call(call.group(0)) != [{"function": name, "arguments": args}]:
        raise ValueError("ambiguous tool markup inside arguments")
    replacement = steps
    if folds:
        replacement = fold_suffix(steps, json.loads(folds[0]))
    elif steps and name != "finish":
        raise ValueError("compress a suffix before the next tool call")
    # Only the isolated call reaches the environment's permissive tool parser.
    return call.group(0), replacement


def _get(value):
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value[0]


def _fit_observation(text, tokenizer, allowance):
    """Explicitly mark any budget truncation; never clip the fixed task prompt."""
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    if len(encode(text)) <= allowance:
        return text
    marker = "\n[Observation truncated by response budget.]"
    if len(encode(marker)) > allowance:
        return ""
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(encode(text[:mid] + marker)) <= allowance:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + marker


async def process_item(item, context):
    if context.is_train:
        raise ValueError("AgentFold adaptation is evaluation-only; no training trajectory export")
    ability = _get(item.non_tensor_batch["ability"])
    if ability not in {"LocalSearch", "GAIA"}:
        raise ValueError("AgentFold adaptation currently supports LocalSearch and text-only GAIA")
    config = copy.deepcopy(context.config.actor_rollout_ref.rollout)
    plugin, tokenizer = config.plugin, context.tokenizer
    budget = int(getattr(plugin, "val_response_length", None) or config.response_length)
    window = int(config.prompt_length) + budget
    max_turn = int(getattr(plugin, "val_max_turn", plugin.max_turn))
    cap = int(getattr(plugin, "turn_max_new_tokens", 2048))
    reserve = max(0, int(getattr(plugin, "final_answer_reserve", 0)))
    margin = max(0, int(getattr(plugin, "final_answer_safety_margin", 64)))
    deadline = time.monotonic() + float(getattr(plugin, "session_timeout", 3600))
    env = select_env(ability, config)(config, tokenizer, ability)
    env.raise_judge_errors = True
    async with managed_environment(env):
        await env.init_env(item)
        fixed = create_chat(env.instance_info["problem_statement"], "search_single", item)
        fixed[0]["content"] += "\n\n" + FOLD_PROMPT
        transcript = copy.deepcopy(fixed)
        steps, audit = [], []
        stats = {"environment_steps": 0, "agentfold_folds": 0, "agentfold_deep_folds": 0,
                 "invalid_tool": 0, "generated_tokens": 0, "observation_tokens": 0, "feedback_tokens": 0,
                 "observation_budget_truncations": 0, "observation_budget_skips": 0}
        used, iteration, next_step = 0, 0, 1
        feedback, rejected, reason = "", "", "max_turn"
        consecutive_invalid = 0
        input_ids, output_ids, logprobs = [], [], []
        while iteration < max_turn:
            remaining = budget - used
            finish_only = bool(reserve and (remaining <= reserve + margin + max(cap, 10)
                                           or iteration == max_turn - 1))
            messages = copy.deepcopy(fixed)
            history = "\n\n".join(step.render() for step in steps)
            messages[-1]["content"] += "\n\nWorking history:\n" + (history or "(No completed steps.)")
            if feedback:
                messages.extend([{"role": "assistant", "content": rejected},
                                 {"role": "user", "content": feedback}])
            if finish_only:
                messages[-1]["content"] += "\nBudget ending: submit one finish call now; no compression needed."
            candidate_ids = list(_apply_chat_template(tokenizer, messages, config,
                                 tokenize=True, add_generation_prompt=True))
            limit = min(window - len(candidate_ids), remaining - (reserve + margin if reserve and not finish_only else 0))
            if cap > 0:
                limit = min(limit, cap)
            if limit < 10:
                reason = "token_limit"
                break
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                reason = "timeout"
                break
            iteration += 1
            try:
                result = await asyncio.wait_for(context.llm_client.create_completion(
                    candidate_ids, messages=messages, max_new_tokens=limit,
                    max_len=len(candidate_ids) + limit), timeout=timeout)
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
                transcript.append({"role": "assistant", "content": text})
                record = {"input_ids": input_ids, "output_ids": output_ids, "messages": messages,
                          "max_tokens": limit, "response": text}
                audit.append(record)
                try:
                    action, replacement = parse_response(text, steps, finish_only=finish_only)
                except (ValueError, TypeError) as exc:
                    stats["invalid_tool"] += 1
                    record["format_error"] = str(exc)
                    consecutive_invalid += 1
                    if consecutive_invalid >= MAX_FORMAT_ERRORS:
                        reason = "invalid_tool_limit"
                        break
                    rejected, correction = retry_context(text, str(exc), tokenizer)
                    feedback = _fit_observation(correction, tokenizer, max(0, budget - used))
                    cost = len(tokenizer.encode(feedback, add_special_tokens=False))
                    used += cost
                    stats["feedback_tokens"] += cost
                    continue
                feedback, rejected, consecutive_invalid = "", "", 0
                if replacement is not steps:
                    stats["agentfold_folds"] += 1
                    stats["agentfold_deep_folds"] += int(replacement[-1].start != replacement[-1].end)
                steps = replacement
                record["action"] = action
                result = await asyncio.wait_for(env.run_action(action),
                    timeout=max(0, deadline - time.monotonic()))
                stats["environment_steps"] += 1
                if getattr(env, "env_fail", False):
                    raise RuntimeError("AgentFold environment failed")
                if getattr(env, "is_finish", False):
                    reason = "finish"
                    break
                raw = str(result.get("observation", "Empty"))
                record["raw_observation"] = raw
                allowance = max(0, budget - used - reserve - margin)
                shown = _fit_observation(raw, tokenizer, allowance)
                stats["observation_budget_truncations"] += int(bool(shown) and shown != raw)
                stats["observation_budget_skips"] += int(not shown)
                cost = len(tokenizer.encode(shown, add_special_tokens=False))
                used += cost
                stats["observation_tokens"] += cost
                transcript.append({"role": "user", "content": shown})
                record["shown_observation"] = shown
                # Keep reasoning/action, but do not duplicate the submitted compression text.
                interaction = re.sub(r"<compress>.*?</compress>", "", text, flags=re.S)
                steps.append(Step(next_step, next_step, interaction + "\nObservation:\n" + shown))
                next_step += 1
            except asyncio.TimeoutError:
                reason = "timeout"
                break
        is_finish = bool(getattr(env, "is_finish", False))
        _, reward, _ = await asyncio.wait_for(env.get_reward(item, transcript, context), timeout=600)
        stats.update(dict(getattr(env, "stats", {})))
        stats.update(task_reward=float(reward), main_turn=iteration, main_len=used,
                     session_time=time.monotonic() - (deadline - float(getattr(plugin, "session_timeout", 3600))),
                     model_requests=len(audit),
                     total_token=sum(len(r["input_ids"]) + len(r["output_ids"]) for r in audit),
                     working_context_limit=window, main_context_tokens=len(input_ids),
                     hit_token_limit=int(reason == "token_limit"), hit_max_turn=int(reason == "max_turn"),
                     hit_timeout=int(reason == "timeout"))
        stats["hit_format_retry_limit"] = int(reason == "invalid_tool_limit")
        return AgentLoopOutput(prompt_ids=input_ids, response_ids=output_ids,
            response_mask=[0] * len(output_ids), response_logprobs=logprobs,
            reward_score=float(reward), num_turns=iteration, metrics=AgentLoopMetrics(),
            extra_fields={"messages": transcript, "model_contexts": audit, "env_stats": stats,
                          "is_finish": is_finish, "termination_reason": reason,
                          "judge_audit": copy.deepcopy(getattr(env, "judge_audit", [])),
                          "agentfold": PROVENANCE, "working_history": [s.__dict__ for s in steps]})
