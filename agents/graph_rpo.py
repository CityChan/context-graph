"""GraphRPO graph-potential evaluation and edit-local credit assignment.

The policy and graph evaluator are intentionally separated.  During rollout we
record the policy-facing graph view immediately before and after every graph
decision.  Once the verified terminal outcome is known, this module batches
those frozen views to an external, frozen evaluator and broadcasts each valid
edit's bounded utility increment over the tokens of its structured decision.

The counterfactual QA backend estimates graph utility with paired downstream
answer rollouts from the pre-update policy.  For each graph edit it samples
answers from the before/after graph views with matched seeds, scores those
answers with the task's normal correctness judge, and credits the edit with
the bounded difference in mean task reward.  Probe tokens are never returned
as training trajectories.

The answer-likelihood alternatives record the same edit-local spans and let
the PPO driver score the known correct answer under either the frozen
reference policy or a no-grad snapshot of the current policy before its
optimizer step. Keeping that scoring on the trainer side avoids loading a
second learned evaluator in the rollout workers.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import aiohttp

from .graph_trace import semantic_snapshot_hash


GRAPH_EVALUATOR_SCHEMA_VERSION = "contextgraph.graph_evaluator.v1"
STATE_CHANGING_GRAPH_OPS = frozenset({"merge", "prune", "add_edge", "select"})
EXTERNAL_EVALUATOR_BACKEND = "external_evaluator"
REFERENCE_ANSWER_LIKELIHOOD_BACKEND = "reference_answer_likelihood"
OLD_POLICY_ANSWER_LIKELIHOOD_BACKEND = "old_policy_answer_likelihood"
OLD_POLICY_COUNTERFACTUAL_QA_BACKEND = "old_policy_counterfactual_qa"
ANSWER_LIKELIHOOD_BACKENDS = frozenset(
    {REFERENCE_ANSWER_LIKELIHOOD_BACKEND, OLD_POLICY_ANSWER_LIKELIHOOD_BACKEND}
)
GRAPH_RPO_CREDIT_BACKENDS = frozenset(
    {
        "evidence",
        EXTERNAL_EVALUATOR_BACKEND,
        OLD_POLICY_COUNTERFACTUAL_QA_BACKEND,
        *ANSWER_LIKELIHOOD_BACKENDS,
    }
)
COUNTERFACTUAL_QA_PROMPT_VERSION = "contextgraph.counterfactual_qa.v1"


class GraphRPOEvaluatorError(RuntimeError):
    """Raised when the frozen graph evaluator cannot provide valid scores."""


def format_graph_evaluator_input(question: str, graph_view: str) -> str:
    """Use one canonical evaluator prompt for both training and serving."""
    return f"Question:\n{question.strip()}\n\nContextGraph view:\n{graph_view.strip()}"


def _config_get(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _token_count(tokenizer: Any, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def _scale_and_clip_delta(
    raw_delta: float,
    *,
    delta_scale: float,
    delta_max: float,
) -> tuple[float, float]:
    """Return the calibrated pre-clip delta and its bounded training value."""
    if not math.isfinite(delta_scale) or delta_scale <= 0.0:
        raise ValueError("graph_rpo_delta_scale must be positive")
    if not math.isfinite(delta_max) or delta_max <= 0.0:
        raise ValueError("graph_rpo_delta_max must be positive")
    scaled_delta = float(raw_delta) / delta_scale
    return scaled_delta, min(max(scaled_delta, -delta_max), delta_max)


def graph_rpo_credit_backend(plugin_config: Any) -> str:
    """Return and validate the configured graph-credit backend."""
    backend = str(
        _config_get(
            plugin_config,
            "graph_rpo_credit_backend",
            OLD_POLICY_COUNTERFACTUAL_QA_BACKEND,
        )
        or OLD_POLICY_COUNTERFACTUAL_QA_BACKEND
    ).strip().lower()
    if backend not in GRAPH_RPO_CREDIT_BACKENDS:
        choices = ", ".join(sorted(GRAPH_RPO_CREDIT_BACKENDS))
        raise ValueError(f"graph_rpo_credit_backend must be one of {choices}; got {backend!r}")
    return backend


def valid_graph_edit_events(graph_trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Select successful model-authored edits that changed graph state."""
    return [
        event
        for event in graph_trace.get("events", [])
        if event.get("source") == "model"
        and event.get("success") is True
        and str(event.get("op", "")).lower() in STATE_CHANGING_GRAPH_OPS
        and _event_has_semantic_state_change(event)
    ]


def _event_has_semantic_state_change(event: Mapping[str, Any]) -> bool:
    """Prefer counter-free graph hashes while accepting legacy minimal traces."""
    before_semantic = event.get("semantic_before_hash")
    after_semantic = event.get("semantic_after_hash")
    if isinstance(before_semantic, str) and isinstance(after_semantic, str):
        return before_semantic != after_semantic

    before_state = event.get("before_state")
    after_state = event.get("after_state")
    if isinstance(before_state, dict) and isinstance(after_state, dict):
        return semantic_snapshot_hash(before_state) != semantic_snapshot_hash(after_state)

    # Compatibility for old unit fixtures or compact traces without snapshots.
    return event.get("before_hash") != event.get("after_hash")


def format_counterfactual_qa_messages(
    question: str, graph_view: str
) -> list[dict[str, str]]:
    """Build a tool-free downstream QA probe conditioned on one graph state."""
    return [
        {
            "role": "system",
            "content": (
                "Answer the question using the supplied ContextGraph memory. "
                "Do not search or call tools. Give your final answer inside exactly one "
                "<answer>...</answer> tag. For multi-part questions, preserve the requested "
                "<q1>...</q1> answer format inside that tag."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question.strip()}\n\n"
                f"ContextGraph memory:\n{graph_view.strip()}"
            ),
        },
    ]


def extract_counterfactual_answer(response: str) -> str:
    """Extract the policy's submitted answer while retaining a robust fallback."""
    text = str(response or "").strip()
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    match = re.search(
        r"(?:^|\n)\s*(?:final\s+answer|answer)\s*[:=]\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip().rstrip(".")
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip().rstrip(".")
    return text


def counterfactual_answer_format_valid(response: str) -> bool:
    """Return whether a probe obeyed the single ``<answer>`` tag contract."""
    matches = re.findall(
        r"<answer>(.*?)</answer>",
        str(response or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return len(matches) == 1 and bool(matches[0].strip())


async def assign_counterfactual_graph_edit_credits(
    *,
    agent: Any,
    graph_trace: dict[str, Any],
    question: str,
    generate_answer: Callable[[str, int, int], Awaitable[str]],
    score_answer: Callable[[str, list[dict[str, Any]]], Awaitable[float]],
    plugin_config: Any,
) -> dict[str, float | int]:
    """Credit edits by paired pre-update-policy QA outcomes on before/after views.

    This estimator deliberately does not gate on the reward of the original
    episode: an edit can receive positive or negative local credit even when
    the main trajectory ultimately answered incorrectly.
    """
    events = valid_graph_edit_events(graph_trace)
    metrics: dict[str, float | int] = {
        "graph_rpo_valid_edits": len(events),
        "graph_rpo_creditable_edits": 0,
        "graph_rpo_credited_edits": 0,
        "graph_rpo_scored_states": 0,
        "graph_rpo_delta_sum": 0.0,
        "graph_rpo_delta_abs_sum": 0.0,
        "graph_rpo_counterfactual_scored_states": 0,
        "graph_rpo_counterfactual_probe_rollouts": 0,
        "graph_rpo_counterfactual_tagged_responses": 0,
        "graph_rpo_counterfactual_tag_rate": 0.0,
        "graph_rpo_counterfactual_positive_rewards": 0,
        "graph_rpo_counterfactual_positive_rate": 0.0,
        "graph_rpo_counterfactual_nonzero_edits": 0,
        "graph_rpo_counterfactual_delta_sum": 0.0,
        "graph_rpo_counterfactual_delta_abs_sum": 0.0,
    }
    if not events:
        return metrics

    num_samples = int(
        _config_get(plugin_config, "graph_rpo_counterfactual_samples", 2)
    )
    delta_scale = float(_config_get(plugin_config, "graph_rpo_delta_scale", 1.0))
    delta_max = float(_config_get(plugin_config, "graph_rpo_delta_max", 1.0))
    base_seed = int(_config_get(plugin_config, "graph_rpo_counterfactual_seed", 42))
    raw_costs = _config_get(plugin_config, "graph_rpo_operation_costs", {}) or {}
    operation_costs = {
        str(key).lower(): float(value) for key, value in dict(raw_costs).items()
    }
    if num_samples <= 0:
        raise ValueError("graph_rpo_counterfactual_samples must be positive")
    _scale_and_clip_delta(0.0, delta_scale=delta_scale, delta_max=delta_max)
    if any(value < 0.0 for value in operation_costs.values()):
        raise ValueError("GraphRPO operation costs must be non-negative")

    unique_views: list[str] = []
    for event in events:
        for key in ("rendered_before", "rendered_after"):
            view = event.get(key)
            if not isinstance(view, str) or not view:
                raise GraphRPOEvaluatorError(
                    f"graph trace event {event.get('seq')} is missing {key}"
                )
            if view not in unique_views:
                unique_views.append(view)

    question_seed = int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16)
    view_results: dict[str, dict[str, Any]] = {}
    for view in unique_views:
        seeds: list[int] = []
        responses: list[str] = []
        answers: list[str] = []
        rewards: list[float] = []
        format_valid: list[bool] = []
        audits: list[list[dict[str, Any]]] = []
        for sample_index in range(num_samples):
            # Common random numbers reduce variance: the paired before/after
            # probes use the same seed for each sample index.
            seed = (base_seed + question_seed + sample_index) % (2**31)
            response = await generate_answer(view, sample_index, seed)
            answer = extract_counterfactual_answer(response)
            response_format_valid = counterfactual_answer_format_valid(response)
            judge_audit: list[dict[str, Any]] = []
            if response_format_valid:
                reward = float(await score_answer(answer, judge_audit))
            else:
                reward = 0.0
                judge_audit.append(
                    {
                        "score": 0.0,
                        "judge_method": "counterfactual_format_invalid",
                        "parse_error": True,
                    }
                )
            if not math.isfinite(reward):
                raise GraphRPOEvaluatorError("counterfactual task reward must be finite")
            if not 0.0 <= reward <= 1.0:
                raise GraphRPOEvaluatorError(
                    "counterfactual task reward must be in [0, 1]"
                )
            seeds.append(seed)
            responses.append(response)
            answers.append(answer)
            rewards.append(reward)
            format_valid.append(response_format_valid)
            audits.append(judge_audit)
        view_results[view] = {
            "seeds": seeds,
            "responses": responses,
            "answers": answers,
            "rewards": rewards,
            "format_valid": format_valid,
            "judge_audits": audits,
            "utility": sum(rewards) / len(rewards),
        }

    metrics["graph_rpo_counterfactual_scored_states"] = len(unique_views)
    metrics["graph_rpo_scored_states"] = len(unique_views)
    metrics["graph_rpo_counterfactual_probe_rollouts"] = len(unique_views) * num_samples
    all_probe_results = [
        (format_valid, reward)
        for result in view_results.values()
        for format_valid, reward in zip(result["format_valid"], result["rewards"])
    ]
    tagged_responses = sum(int(format_valid) for format_valid, _ in all_probe_results)
    positive_rewards = sum(int(reward > 0.0) for _, reward in all_probe_results)
    metrics["graph_rpo_counterfactual_tagged_responses"] = tagged_responses
    metrics["graph_rpo_counterfactual_positive_rewards"] = positive_rewards
    metrics["graph_rpo_counterfactual_tag_rate"] = (
        tagged_responses / len(all_probe_results) if all_probe_results else 0.0
    )
    metrics["graph_rpo_counterfactual_positive_rate"] = (
        positive_rewards / len(all_probe_results) if all_probe_results else 0.0
    )
    for event in events:
        before = view_results[event["rendered_before"]]
        after = view_results[event["rendered_after"]]
        op = str(event["op"]).lower()
        operation_cost = operation_costs.get(op, 0.0)
        raw_delta = float(after["utility"]) - float(before["utility"]) - operation_cost
        scaled_delta, delta = _scale_and_clip_delta(
            raw_delta,
            delta_scale=delta_scale,
            delta_max=delta_max,
        )
        turn_index = event.get("assistant_turn_index")
        if not isinstance(turn_index, int):
            raise GraphRPOEvaluatorError(
                f"graph trace event {event.get('seq')} lacks an assistant turn index"
            )
        agent.add_graph_edit_credit(turn_index, delta)
        event.update(
            {
                "graph_rpo_credit_backend": OLD_POLICY_COUNTERFACTUAL_QA_BACKEND,
                "graph_rpo_counterfactual_prompt_version": COUNTERFACTUAL_QA_PROMPT_VERSION,
                "graph_rpo_counterfactual_samples": num_samples,
                "graph_rpo_counterfactual_seeds": before["seeds"],
                "graph_rpo_counterfactual_before_responses": before["responses"],
                "graph_rpo_counterfactual_after_responses": after["responses"],
                "graph_rpo_counterfactual_before_answers": before["answers"],
                "graph_rpo_counterfactual_after_answers": after["answers"],
                "graph_rpo_counterfactual_before_rewards": before["rewards"],
                "graph_rpo_counterfactual_after_rewards": after["rewards"],
                "graph_rpo_counterfactual_before_format_valid": before["format_valid"],
                "graph_rpo_counterfactual_after_format_valid": after["format_valid"],
                "graph_rpo_counterfactual_before_judge_audits": before["judge_audits"],
                "graph_rpo_counterfactual_after_judge_audits": after["judge_audits"],
                "graph_rpo_utility_before": before["utility"],
                "graph_rpo_utility_after": after["utility"],
                "graph_rpo_operation_cost": operation_cost,
                "graph_rpo_delta_unclipped": raw_delta,
                "graph_rpo_delta_scale": delta_scale,
                "graph_rpo_delta_scaled_unclipped": scaled_delta,
                "graph_rpo_delta": delta,
                "graph_rpo_outcome_gated": False,
            }
        )
        metrics["graph_rpo_creditable_edits"] = int(metrics["graph_rpo_creditable_edits"]) + 1
        metrics["graph_rpo_counterfactual_delta_sum"] = (
            float(metrics["graph_rpo_counterfactual_delta_sum"]) + delta
        )
        metrics["graph_rpo_counterfactual_delta_abs_sum"] = (
            float(metrics["graph_rpo_counterfactual_delta_abs_sum"]) + abs(delta)
        )
        if abs(delta) > 1e-12:
            metrics["graph_rpo_credited_edits"] = (
                int(metrics["graph_rpo_credited_edits"]) + 1
            )
            metrics["graph_rpo_counterfactual_nonzero_edits"] = (
                int(metrics["graph_rpo_counterfactual_nonzero_edits"]) + 1
            )
        metrics["graph_rpo_delta_sum"] = float(metrics["graph_rpo_delta_sum"]) + delta
        metrics["graph_rpo_delta_abs_sum"] = (
            float(metrics["graph_rpo_delta_abs_sum"]) + abs(delta)
        )
    return metrics


def prepare_reference_graph_edit_requests(
    *,
    graph_trace: dict[str, Any],
    terminal_reward: float,
    credit_backend: str = REFERENCE_ANSWER_LIKELIHOOD_BACKEND,
) -> tuple[list[dict[str, int]], dict[str, float | int]]:
    """Record edit identities for answer-likelihood scoring in the PPO driver.

    The large before/after views already live in ``graph_trace``.  Requests
    therefore carry only an event sequence number and assistant-turn index;
    the agent loop adds exact response-token indices after tokenization.
    """
    credit_backend = str(credit_backend).strip().lower()
    if credit_backend not in ANSWER_LIKELIHOOD_BACKENDS:
        choices = ", ".join(sorted(ANSWER_LIKELIHOOD_BACKENDS))
        raise ValueError(f"answer-likelihood backend must be one of {choices}")

    events = valid_graph_edit_events(graph_trace)
    metrics: dict[str, float | int] = {
        "graph_rpo_valid_edits": len(events),
        "graph_rpo_creditable_edits": 0,
        "graph_rpo_credited_edits": 0,
    }
    if not events:
        return [], metrics

    if float(terminal_reward) <= 0.0:
        for event in events:
            event["graph_rpo_delta"] = 0.0
            event["graph_rpo_outcome_gated"] = True
            event["graph_rpo_credit_backend"] = credit_backend
        return [], metrics

    requests: list[dict[str, int]] = []
    for event in events:
        for key in ("rendered_before", "rendered_after"):
            view = event.get(key)
            if not isinstance(view, str) or not view:
                raise GraphRPOEvaluatorError(
                    f"graph trace event {event.get('seq')} is missing {key}"
                )
        turn_index = event.get("assistant_turn_index")
        if not isinstance(turn_index, int):
            raise GraphRPOEvaluatorError(
                f"graph trace event {event.get('seq')} lacks an assistant turn index"
            )
        seq = event.get("seq")
        if not isinstance(seq, int):
            raise GraphRPOEvaluatorError("graph trace edit lacks an integer sequence number")
        event["graph_rpo_credit_backend"] = credit_backend
        event["graph_rpo_outcome_gated"] = False
        requests.append({"seq": seq, "assistant_turn_index": turn_index})
    return requests, metrics


def _parse_probabilities(payload: Any, expected: int) -> list[float]:
    values: Any = None
    if isinstance(payload, Mapping):
        values = payload.get("probabilities", payload.get("scores"))
        if values is None and isinstance(payload.get("outputs"), list):
            values = [item.get("probability") for item in payload["outputs"]]
    elif isinstance(payload, list):
        values = payload

    if not isinstance(values, list) or len(values) != expected:
        raise GraphRPOEvaluatorError(
            "graph evaluator must return one probability per requested graph view"
        )

    probabilities: list[float] = []
    for index, value in enumerate(values):
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise GraphRPOEvaluatorError(
                f"graph evaluator probability {index} is not numeric: {value!r}"
            ) from exc
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise GraphRPOEvaluatorError(
                f"graph evaluator probability {index} must be finite and in [0, 1]"
            )
        probabilities.append(probability)
    return probabilities


async def score_graph_views(
    *,
    endpoint: str,
    question: str,
    graph_views: list[str],
    graph_view_tokens: list[int],
    timeout_seconds: float = 120.0,
) -> list[float]:
    """Score graph views with the configured frozen evaluator service.

    Endpoint contract: POST a ``contextgraph.graph_evaluator.v1`` request and
    return either ``{"probabilities": [...]}``, ``{"scores": [...]}``, or
    ``{"outputs": [{"probability": ...}, ...]}`` in request order.
    """
    endpoint = str(endpoint or "").strip()
    if not endpoint:
        raise GraphRPOEvaluatorError(
            "GraphRPO requires actor_rollout_ref.rollout.plugin."
            "graph_rpo_evaluator_url"
        )
    if len(graph_views) != len(graph_view_tokens):
        raise ValueError("graph views and token counts must have equal length")
    if not graph_views:
        return []

    request = {
        "schema_version": GRAPH_EVALUATOR_SCHEMA_VERSION,
        "items": [
            {
                "question": question,
                "graph_view": view,
                "graph_view_tokens": int(token_count),
            }
            for view, token_count in zip(graph_views, graph_view_tokens, strict=True)
        ],
    }
    timeout = aiohttp.ClientTimeout(total=float(timeout_seconds))
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=request) as response:
                response.raise_for_status()
                payload = await response.json()
    except Exception as exc:
        raise GraphRPOEvaluatorError(
            f"frozen graph evaluator request failed: {exc}"
        ) from exc
    return _parse_probabilities(payload, len(graph_views))


def graph_utility(
    *,
    probability: float,
    serialized_tokens: int,
    budget_tokens: int,
    probability_epsilon: float,
    confidence_bound: float,
    serialization_penalty: float,
) -> float:
    """Compute the bounded evaluator logit minus normalized view footprint."""
    if budget_tokens <= 0:
        raise ValueError("graph_rpo_view_budget must be positive")
    if not 0.0 < probability_epsilon < 0.5:
        raise ValueError("graph_rpo_probability_epsilon must be in (0, 0.5)")
    if confidence_bound <= 0.0:
        raise ValueError("graph_rpo_confidence_bound must be positive")
    if serialization_penalty < 0.0:
        raise ValueError("graph_rpo_serialization_penalty must be non-negative")

    probability = min(max(float(probability), probability_epsilon), 1.0 - probability_epsilon)
    logit = math.log(probability / (1.0 - probability))
    bounded_logit = min(max(logit, -confidence_bound), confidence_bound)
    return bounded_logit - serialization_penalty * float(serialized_tokens) / budget_tokens


async def assign_graph_edit_credits(
    *,
    agent: Any,
    graph_trace: dict[str, Any],
    question: str,
    terminal_reward: float,
    tokenizer: Any,
    plugin_config: Any,
) -> dict[str, float | int]:
    """Evaluate valid model edits and attach their deltas to assistant turns."""
    events = valid_graph_edit_events(graph_trace)
    metrics: dict[str, float | int] = {
        "graph_rpo_valid_edits": len(events),
        "graph_rpo_creditable_edits": 0,
        "graph_rpo_credited_edits": 0,
        "graph_rpo_scored_states": 0,
        "graph_rpo_delta_sum": 0.0,
        "graph_rpo_delta_abs_sum": 0.0,
    }
    if not events:
        return metrics

    # The manuscript gates graph-potential credit on verified task success.
    # Preserve auditable zero deltas without paying for evaluator calls on
    # unsuccessful episodes.
    if float(terminal_reward) <= 0.0:
        for event in events:
            event["graph_rpo_delta"] = 0.0
            event["graph_rpo_outcome_gated"] = True
        return metrics

    endpoint = _config_get(plugin_config, "graph_rpo_evaluator_url", "")
    view_budget = int(_config_get(plugin_config, "graph_rpo_view_budget", 2048))
    epsilon_v = float(_config_get(plugin_config, "graph_rpo_probability_epsilon", 1e-4))
    confidence_bound = float(_config_get(plugin_config, "graph_rpo_confidence_bound", 8.0))
    serialization_penalty = float(_config_get(plugin_config, "graph_rpo_serialization_penalty", 0.0))
    delta_scale = float(_config_get(plugin_config, "graph_rpo_delta_scale", 1.0))
    delta_max = float(_config_get(plugin_config, "graph_rpo_delta_max", 1.0))
    timeout_seconds = float(_config_get(plugin_config, "graph_rpo_evaluator_timeout", 120.0))
    raw_costs = _config_get(plugin_config, "graph_rpo_operation_costs", {}) or {}
    operation_costs = {str(key).lower(): float(value) for key, value in dict(raw_costs).items()}

    _scale_and_clip_delta(0.0, delta_scale=delta_scale, delta_max=delta_max)
    if any(value < 0.0 for value in operation_costs.values()):
        raise ValueError("GraphRPO operation costs must be non-negative")

    unique_views: list[str] = []
    view_to_index: dict[str, int] = {}
    for event in events:
        for key in ("rendered_before", "rendered_after"):
            view = event.get(key)
            if not isinstance(view, str) or not view:
                raise GraphRPOEvaluatorError(
                    f"graph trace event {event.get('seq')} is missing {key}"
                )
            if view not in view_to_index:
                view_to_index[view] = len(unique_views)
                unique_views.append(view)

    token_counts = [_token_count(tokenizer, view) for view in unique_views]
    probabilities = await score_graph_views(
        endpoint=endpoint,
        question=question,
        graph_views=unique_views,
        graph_view_tokens=token_counts,
        timeout_seconds=timeout_seconds,
    )
    utilities = [
        graph_utility(
            probability=probability,
            serialized_tokens=token_count,
            budget_tokens=view_budget,
            probability_epsilon=epsilon_v,
            confidence_bound=confidence_bound,
            serialization_penalty=serialization_penalty,
        )
        for probability, token_count in zip(probabilities, token_counts, strict=True)
    ]
    metrics["graph_rpo_scored_states"] = len(unique_views)
    metrics["graph_rpo_creditable_edits"] = len(events)

    for event in events:
        before_index = view_to_index[event["rendered_before"]]
        after_index = view_to_index[event["rendered_after"]]
        op = str(event["op"]).lower()
        raw_delta = utilities[after_index] - utilities[before_index] - operation_costs.get(op, 0.0)
        scaled_delta, delta = _scale_and_clip_delta(
            raw_delta,
            delta_scale=delta_scale,
            delta_max=delta_max,
        )
        turn_index = event.get("assistant_turn_index")
        if not isinstance(turn_index, int):
            raise GraphRPOEvaluatorError(
                f"graph trace event {event.get('seq')} lacks an assistant turn index"
            )
        agent.add_graph_edit_credit(turn_index, delta)
        event.update(
            {
                "graph_rpo_credit_backend": EXTERNAL_EVALUATOR_BACKEND,
                "graph_rpo_probability_before": probabilities[before_index],
                "graph_rpo_probability_after": probabilities[after_index],
                "graph_rpo_utility_before": utilities[before_index],
                "graph_rpo_utility_after": utilities[after_index],
                "graph_rpo_operation_cost": operation_costs.get(op, 0.0),
                "graph_rpo_delta_unclipped": raw_delta,
                "graph_rpo_delta_scale": delta_scale,
                "graph_rpo_delta_scaled_unclipped": scaled_delta,
                "graph_rpo_delta": delta,
                "graph_rpo_outcome_gated": False,
            }
        )
        metrics["graph_rpo_delta_sum"] = float(metrics["graph_rpo_delta_sum"]) + delta
        metrics["graph_rpo_delta_abs_sum"] = float(metrics["graph_rpo_delta_abs_sum"]) + abs(delta)
        if abs(delta) > 1e-12:
            metrics["graph_rpo_credited_edits"] = (
                int(metrics["graph_rpo_credited_edits"]) + 1
            )
    return metrics
