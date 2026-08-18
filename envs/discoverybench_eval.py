"""DiscoveryBench Hypothesis Match Score (HMS) evaluator.

This follows the upstream evaluator's decomposition and matching procedure:
decompose gold/predicted hypotheses into context-variable-relation facets,
greedily match contexts, score variable overlap and relation fidelity, then
multiply context recall by mean matched accuracy.  The client wrapper adds
Azure OpenAI support required by the Vista environment.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable


class DiscoveryBenchJudgeError(RuntimeError):
    pass


def build_judge_client():
    """Return ``(client, deployment_or_model, provider)``."""
    from openai import AzureOpenAI, OpenAI

    azure_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if azure_key:
        required = {
            "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise DiscoveryBenchJudgeError(
                "Incomplete Azure credentials: missing " + ", ".join(missing)
            )
        client = AzureOpenAI(
            api_key=azure_key,
            api_version=required["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=required["AZURE_OPENAI_ENDPOINT"],
        )
        return client, required["AZURE_OPENAI_DEPLOYMENT_NAME"], "azure"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "dummy":
        raise DiscoveryBenchJudgeError(
            "Official HMS scoring requires OPENAI_API_KEY or the Azure OpenAI credential set"
        )
    model = os.getenv("DISCOVERYBENCH_JUDGE_MODEL", "gpt-4-1106-preview")
    kwargs = {"api_key": api_key}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    return OpenAI(**kwargs), model, "openai"


def _strip_json_fence(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _chat_json(client, model: str, prompt: str, retries: int = 3) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": (
                        "You evaluate data-driven hypotheses. Return only the requested JSON object."
                    )},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 1200,
            }
            try:
                response = client.chat.completions.create(
                    response_format={"type": "json_object"}, **kwargs
                )
            except Exception:
                response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            parsed = json.loads(_strip_json_fence(content))
            if not isinstance(parsed, dict):
                raise ValueError("judge response was not a JSON object")
            return parsed
        except Exception as exc:  # noqa: BLE001 - API retry boundary
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
    raise DiscoveryBenchJudgeError(f"Judge failed after {retries} attempts: {last_error}")


def _metadata_view(metadata: dict, dataset_type: str) -> list[dict]:
    rendered = []
    for dataset in metadata.get("datasets", []):
        columns = dataset.get("columns", [])
        if dataset_type == "real" and isinstance(columns, dict):
            columns = columns.get("raw", [])
        rendered.append({
            "name": dataset.get("name", ""),
            "description": dataset.get("description", ""),
            "columns": [
                {"name": col.get("name", ""), "description": col.get("description", "")}
                for col in columns
            ],
        })
    return rendered


def _decompose(judge: Callable[[str], dict], query: str, hypothesis: str,
               workflow: str, metadata_view: list[dict]) -> list[dict]:
    prompt = f"""Decompose the hypothesis into self-contained sub-hypotheses.
Each sub-hypothesis must have: text, context (boundary conditions; use "None"
only when no boundary is stated), variables (dataset concepts/columns), and
relation (direction, form, threshold, coefficient, or comparison).
Do not add information absent from the hypothesis or workflow.

Dataset metadata: {json.dumps(metadata_view, ensure_ascii=False)}
Question: {query}
Hypothesis: {hypothesis}
Workflow: {workflow}

Return {{"sub_hypotheses": [{{"text": "...", "context": "...",
"variables": ["..."], "relation": "..."}}]}}."""
    result = judge(prompt)
    facets = result.get("sub_hypotheses", result.get("sub_hypo", []))
    if not isinstance(facets, list) or not facets:
        return [{
            "text": hypothesis,
            "context": "None",
            "variables": [],
            "relation": "",
        }]
    normalized = []
    for facet in facets:
        if not isinstance(facet, dict):
            continue
        variables = facet.get("variables", [])
        if not isinstance(variables, list):
            variables = [str(variables)]
        normalized.append({
            "text": str(facet.get("text", hypothesis)),
            "context": str(facet.get("context", "None")),
            "variables": [str(value) for value in variables],
            "relation": str(facet.get("relation", facet.get("relations", ""))),
        })
    return normalized or [{
        "text": hypothesis, "context": "None", "variables": [], "relation": ""
    }]


def _normalized(value: str) -> str:
    return re.sub(r"\W+", " ", value.casefold()).strip()


def _context_match(judge: Callable[[str], dict], gold: dict, pred: dict) -> bool:
    gold_context = _normalized(gold["context"])
    pred_context = _normalized(pred["context"])
    if gold_context == pred_context:
        return True
    if "none" in {gold_context, pred_context}:
        return False
    result = judge(
        "Determine whether the predicted context has the same boundary conditions "
        "as the gold context. Full-dataset contexts match each other even when "
        "paraphrased. Return {\"match\": true|false}.\n"
        f"Gold: {json.dumps(gold, ensure_ascii=False)}\n"
        f"Predicted: {json.dumps(pred, ensure_ascii=False)}"
    )
    return bool(result.get("match", False))


def _variable_f1(judge: Callable[[str], dict], gold: dict, pred: dict) -> tuple[float, dict]:
    result = judge(
        "Compare variables in these two sub-hypotheses using fuzzy semantic "
        "matching for paraphrases. Return non-negative integer counts as "
        "{\"size_gold\": n, \"size_pred\": n, \"intersection\": n}.\n"
        f"Gold: {json.dumps(gold, ensure_ascii=False)}\n"
        f"Predicted: {json.dumps(pred, ensure_ascii=False)}"
    )
    size_gold = max(int(result.get("size_gold", result.get("sizeA", 0))), 0)
    size_pred = max(int(result.get("size_pred", result.get("sizeB", 0))), 0)
    intersection = max(int(result.get("intersection", 0)), 0)
    intersection = min(intersection, size_gold, size_pred)
    precision = intersection / size_pred if size_pred else 0.0
    recall = intersection / size_gold if size_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    result.update({"precision": precision, "recall": recall, "f1": f1})
    return f1, result


def _relation_score(judge: Callable[[str], dict], gold: dict, pred: dict) -> tuple[float, dict]:
    result = judge(
        "Compare the relationship in the predicted sub-hypothesis with the gold. "
        "Use 1.0 for very similar (including quantitative form), 0.5 when the "
        "prediction is compatible but more general, and 0.0 when different or "
        "contradictory. Return {\"score\": 1.0|0.5|0.0, \"explanation\": \"...\"}.\n"
        f"Gold: {json.dumps(gold, ensure_ascii=False)}\n"
        f"Predicted: {json.dumps(pred, ensure_ascii=False)}"
    )
    score = float(result.get("score", 0.0))
    if score not in {0.0, 0.5, 1.0}:
        score = min(max(score, 0.0), 1.0)
    return score, result


def score_hypothesis(
    *,
    query: str,
    gold_hypothesis: str,
    gold_workflow: str,
    predicted_hypothesis: str,
    predicted_workflow: str,
    metadata: dict,
    dataset_type: str,
    client=None,
    model: str | None = None,
) -> dict[str, Any]:
    """Compute an HMS record compatible with the official score definition."""
    if _normalized(gold_hypothesis) == _normalized(predicted_hypothesis):
        return {
            "final_score": 1.0,
            "recall_context": 1.0,
            "mean_accuracy_score": 1.0,
            "exact_match_shortcut": True,
            "judge_model": "not-called",
        }

    provider = "injected"
    if client is None:
        client, discovered_model, provider = build_judge_client()
        model = model or discovered_model
    if not model:
        model = os.getenv("DISCOVERYBENCH_JUDGE_MODEL", "gpt-4-1106-preview")
    judge = lambda prompt: _chat_json(client, model, prompt)  # noqa: E731
    metadata_view = _metadata_view(metadata, dataset_type)
    gold_facets = _decompose(
        judge, query, gold_hypothesis, gold_workflow, metadata_view
    )
    pred_facets = _decompose(
        judge, query, predicted_hypothesis, predicted_workflow, metadata_view
    )

    covered_gold: set[int] = set()
    matches = []
    for pred_index, pred in enumerate(pred_facets):
        for gold_index, gold in enumerate(gold_facets):
            if gold_index in covered_gold:
                continue
            if _context_match(judge, gold, pred):
                covered_gold.add(gold_index)
                variable_f1, variable_record = _variable_f1(judge, gold, pred)
                relation, relation_record = _relation_score(judge, gold, pred)
                matches.append({
                    "pred_index": pred_index,
                    "gold_index": gold_index,
                    "context_score": 1.0,
                    "variable": variable_record,
                    "relation": relation_record,
                    "accuracy_score": variable_f1 * relation,
                })
                break

    recall_context = len(covered_gold) / len(gold_facets) if gold_facets else 0.0
    # Upstream averages over all predicted facets, including unmatched ones.
    mean_accuracy = (
        sum(match["accuracy_score"] for match in matches) / len(pred_facets)
        if pred_facets else 0.0
    )
    return {
        "final_score": recall_context * mean_accuracy,
        "recall_context": recall_context,
        "mean_accuracy_score": mean_accuracy,
        "gold_sub_hypotheses": gold_facets,
        "predicted_sub_hypotheses": pred_facets,
        "matches": matches,
        "judge_model": model,
        "judge_provider": provider,
        "exact_match_shortcut": False,
    }
