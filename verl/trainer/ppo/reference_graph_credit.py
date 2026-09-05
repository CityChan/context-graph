"""Frozen-reference answer-likelihood credit for GraphRPO graph edits.

This module intentionally contains no model lifecycle logic.  It prepares
teacher-forced ``(graph view, correct answer)`` examples for the reference
worker already owned by the PPO trainer, and maps the resulting answer
log-likelihood differences back to the policy tokens that emitted each edit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


REFERENCE_SYSTEM_PROMPT = (
    "Use the supplied ContextGraph evidence to answer the question. "
    "Return only the short final answer."
)


@dataclass(frozen=True)
class ReferenceViewKey:
    question: str
    answer: str
    graph_view: str


@dataclass
class ReferenceEditPlan:
    row_index: int
    event: dict[str, Any]
    before_key: ReferenceViewKey
    after_key: ReferenceViewKey
    response_token_indices: list[int]


def format_reference_answer_prompt(question: str, graph_view: str) -> list[dict[str, str]]:
    """Build the canonical prompt used to score a known answer.

    The question is placed after the graph so left truncation retains both the
    assistant generation marker and the task being answered.
    """
    user_content = (
        f"ContextGraph view:\n{graph_view.strip()}\n\n"
        f"Question:\n{question.strip()}\n\nShort final answer:"
    )
    return [
        {"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def normalize_reference_answer(value: Any) -> str:
    """Normalize the benchmark answer into one non-empty teacher-forced target."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if str(item).strip()), "")
    answer = str(value if value is not None else "").strip()
    if not answer:
        raise ValueError("reference-answer GraphRPO requires a non-empty correct answer")
    return answer


def collect_reference_edit_plans(
    batch: DataProto,
) -> tuple[list[ReferenceEditPlan], list[ReferenceViewKey]]:
    """Resolve rollout edit requests into deduplicated graph-view score items."""
    plans: list[ReferenceEditPlan] = []
    unique_keys: list[ReferenceViewKey] = []
    seen_keys: set[ReferenceViewKey] = set()
    response_length = int(batch.batch["responses"].shape[-1])
    dummy_gen_uid = batch.meta_info.get("gen_uid_dummy")
    gen_uids = batch.non_tensor_batch.get("gen_uid")

    for row_index in range(len(batch)):
        if gen_uids is not None and dummy_gen_uid is not None and gen_uids[row_index] == dummy_gen_uid:
            continue
        if batch.non_tensor_batch.get("agent_name", np.array([None] * len(batch), dtype=object))[row_index] != "main":
            continue
        requests = batch.non_tensor_batch.get("graph_rpo_reference_edits")
        row_requests = requests[row_index] if requests is not None else None
        if not isinstance(row_requests, list) or not row_requests:
            continue

        questions = batch.non_tensor_batch.get("graph_rpo_reference_question")
        answers = batch.non_tensor_batch.get("graph_rpo_reference_answer")
        traces = batch.non_tensor_batch.get("graph_trace")
        if questions is None or answers is None or traces is None:
            raise ValueError("reference-answer GraphRPO rollout metadata is incomplete")
        question = str(questions[row_index] or "").strip()
        if not question:
            raise ValueError("reference-answer GraphRPO requires a non-empty question")
        answer = normalize_reference_answer(answers[row_index])
        trace = traces[row_index]
        if not isinstance(trace, dict):
            raise ValueError("reference-answer GraphRPO requires a graph trace dictionary")
        events_by_seq = {
            event.get("seq"): event
            for event in trace.get("events", [])
            if isinstance(event, dict)
        }

        for request in row_requests:
            if not isinstance(request, dict):
                raise ValueError("reference graph edit request must be a dictionary")
            seq = request.get("seq")
            event = events_by_seq.get(seq)
            if event is None:
                raise ValueError(f"reference graph edit request points to missing event {seq!r}")
            before = event.get("rendered_before")
            after = event.get("rendered_after")
            if not isinstance(before, str) or not before or not isinstance(after, str) or not after:
                raise ValueError(f"reference graph edit event {seq!r} lacks before/after views")
            raw_indices = request.get("response_token_indices", [])
            token_indices = sorted(
                {
                    int(index)
                    for index in raw_indices
                    if isinstance(index, (int, np.integer)) and 0 <= int(index) < response_length
                }
            )
            token_indices = [
                index
                for index in token_indices
                if bool(batch.batch["response_mask"][row_index, index].item())
            ]
            if not token_indices:
                event["graph_rpo_credit_skipped"] = "no_optimized_edit_tokens"
                continue

            before_key = ReferenceViewKey(question, answer, before)
            after_key = ReferenceViewKey(question, answer, after)
            for key in (before_key, after_key):
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_keys.append(key)
            plans.append(
                ReferenceEditPlan(
                    row_index=row_index,
                    event=event,
                    before_key=before_key,
                    after_key=after_key,
                    response_token_indices=token_indices,
                )
            )
    return plans, unique_keys


def _encode_prompt(
    tokenizer: Any,
    key: ReferenceViewKey,
    max_prompt_length: int,
    enable_thinking: bool,
) -> list[int]:
    messages = format_reference_answer_prompt(key.question, key.graph_view)
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except (AttributeError, TypeError, ValueError):
        text = (
            f"{REFERENCE_SYSTEM_PROMPT}\n\n"
            f"{messages[-1]['content']}\n"
        )
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_ids = [int(token_id) for token_id in prompt_ids]
    if not prompt_ids:
        raise ValueError("reference-answer prompt tokenization produced no tokens")
    return prompt_ids[-max_prompt_length:]


def _encode_answer(tokenizer: Any, answer: str, max_answer_length: int) -> list[int]:
    try:
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    except TypeError:
        answer_ids = tokenizer.encode(answer)
    if hasattr(answer_ids, "tolist"):
        answer_ids = answer_ids.tolist()
    answer_ids = [int(token_id) for token_id in answer_ids][:max_answer_length]
    if not answer_ids:
        raise ValueError("reference answer tokenization produced no tokens")
    return answer_ids


def build_reference_scoring_batch(
    tokenizer: Any,
    keys: list[ReferenceViewKey],
    *,
    max_prompt_length: int,
    max_answer_length: int,
    enable_thinking: bool = False,
) -> tuple[DataProto, torch.Tensor]:
    """Build a causal-LM batch whose response is exactly the correct answer."""
    if not keys:
        raise ValueError("cannot build an empty reference-answer scoring batch")
    if max_prompt_length <= 0 or max_answer_length <= 0:
        raise ValueError("reference prompt and answer limits must be positive")

    prompt_rows = [
        _encode_prompt(tokenizer, key, max_prompt_length, enable_thinking)
        for key in keys
    ]
    answer_rows = [_encode_answer(tokenizer, key.answer, max_answer_length) for key in keys]
    prompt_width = max(len(row) for row in prompt_rows)
    answer_width = max(len(row) for row in answer_rows)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("reference-answer scoring requires a tokenizer pad or EOS token")

    prompts = torch.full((len(keys), prompt_width), int(pad_token_id), dtype=torch.long)
    responses = torch.full((len(keys), answer_width), int(pad_token_id), dtype=torch.long)
    prompt_mask = torch.zeros_like(prompts)
    response_mask = torch.zeros_like(responses)
    for row_index, (prompt_ids, answer_ids) in enumerate(zip(prompt_rows, answer_rows, strict=True)):
        prompts[row_index, -len(prompt_ids):] = torch.tensor(prompt_ids, dtype=torch.long)
        prompt_mask[row_index, -len(prompt_ids):] = 1
        responses[row_index, :len(answer_ids)] = torch.tensor(answer_ids, dtype=torch.long)
        response_mask[row_index, :len(answer_ids)] = 1

    input_ids = torch.cat((prompts, responses), dim=-1)
    attention_mask = torch.cat((prompt_mask, response_mask), dim=-1)
    scoring_batch = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "response_mask": response_mask,
        },
        meta_info={"ref_log_prob_temperature": 1.0},
    )
    return scoring_batch, response_mask


def mean_answer_log_likelihood(
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> list[float]:
    """Reduce token log probabilities to length-normalized answer likelihood."""
    if ref_log_prob.shape != response_mask.shape:
        raise ValueError(
            f"reference log-prob shape {tuple(ref_log_prob.shape)} does not match "
            f"answer mask {tuple(response_mask.shape)}"
        )
    mask = response_mask.to(dtype=ref_log_prob.dtype)
    counts = mask.sum(dim=-1)
    if bool((counts <= 0).any().item()):
        raise ValueError("every reference answer must contain at least one token")
    values = (ref_log_prob * mask).sum(dim=-1) / counts
    return [float(value) for value in values.detach().cpu().tolist()]


def apply_reference_edit_credits(
    batch: DataProto,
    plans: list[ReferenceEditPlan],
    likelihoods: dict[ReferenceViewKey, float],
    *,
    delta_max: float,
    operation_costs: dict[str, float] | None = None,
) -> dict[str, float | int]:
    """Write answer-likelihood deltas to edit-token advantages and trace audit data."""
    if delta_max <= 0.0:
        raise ValueError("graph_rpo_delta_max must be positive")
    costs = {str(key).lower(): float(value) for key, value in (operation_costs or {}).items()}
    if any(value < 0.0 for value in costs.values()):
        raise ValueError("GraphRPO operation costs must be non-negative")

    credit_mask = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
    delta_sum = 0.0
    delta_abs_sum = 0.0
    for plan in plans:
        before = float(likelihoods[plan.before_key])
        after = float(likelihoods[plan.after_key])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError("reference-answer GraphRPO received a non-finite likelihood")
        op = str(plan.event.get("op", "")).lower()
        operation_cost = costs.get(op, 0.0)
        raw_delta = after - before - operation_cost
        delta = min(max(raw_delta, -delta_max), delta_max)
        credit_mask[plan.row_index, plan.response_token_indices] += delta
        plan.event.update(
            {
                "graph_rpo_credit_backend": "reference_answer_likelihood",
                "graph_rpo_answer_log_likelihood_before": before,
                "graph_rpo_answer_log_likelihood_after": after,
                "graph_rpo_operation_cost": operation_cost,
                "graph_rpo_delta_unclipped": raw_delta,
                "graph_rpo_delta": delta,
                "graph_rpo_outcome_gated": False,
            }
        )
        delta_sum += delta
        delta_abs_sum += abs(delta)

    batch.batch["graph_edit_credit_mask"] = credit_mask * batch.batch["response_mask"].to(torch.float32)

    # Keep rollout dumps auditable even though reward-manager aggregation has
    # already happened by the time the reference worker returns.
    env_stats = batch.non_tensor_batch.get("env_stats")
    traces = batch.non_tensor_batch.get("graph_trace")
    gen_uids = batch.non_tensor_batch.get("gen_uid")
    per_row_plans: dict[int, list[ReferenceEditPlan]] = {}
    for plan in plans:
        per_row_plans.setdefault(plan.row_index, []).append(plan)
    for row_index, row_plans in per_row_plans.items():
        episode_metrics = {
            "graph_rpo_scored_states": len(
                {plan.before_key for plan in row_plans} | {plan.after_key for plan in row_plans}
            ),
            "graph_rpo_delta_sum": sum(
                float(plan.event["graph_rpo_delta"]) for plan in row_plans
            ),
            "graph_rpo_delta_abs_sum": sum(
                abs(float(plan.event["graph_rpo_delta"])) for plan in row_plans
            ),
        }
        episode_gen_uid = gen_uids[row_index] if gen_uids is not None else None
        audit_by_seq = {
            plan.event.get("seq"): {
                key: value
                for key, value in plan.event.items()
                if key.startswith("graph_rpo_")
            }
            for plan in row_plans
        }
        target_rows = range(len(batch)) if episode_gen_uid is not None else [row_index]
        for target_row in target_rows:
            if gen_uids is not None and gen_uids[target_row] != episode_gen_uid:
                continue
            if env_stats is not None and isinstance(env_stats[target_row], dict):
                env_stats[target_row].update(episode_metrics)
            if traces is not None and isinstance(traces[target_row], dict):
                for event in traces[target_row].get("events", []):
                    if isinstance(event, dict) and event.get("seq") in audit_by_seq:
                        event.update(audit_by_seq[event["seq"]])

    return {
        "graphrpo/reference_creditable_edits": len(plans),
        "graphrpo/reference_scored_states": len(likelihoods),
        "graphrpo/reference_delta_sum": delta_sum,
        "graphrpo/reference_delta_abs_sum": delta_abs_sum,
        "graphrpo/reference_delta_mean": delta_sum / len(plans) if plans else 0.0,
    }
