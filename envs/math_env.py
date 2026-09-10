"""Self-contained exact-match math environment for ContextGraph training."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from collections import Counter
from decimal import Decimal, InvalidOperation


FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
PARAMETER_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
ANSWER_BLOCK_RES = (
    re.compile(r"<parameter=answer>\s*(.*?)\s*</parameter>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL),
)
ANSWER_MARKER_RE = re.compile(
    r"(?:your\s+best\s+answer|final\s+answer|answer)\s*"
    r"(?:(?:is\b)|[>=:])\s*[^\d+\-]{0,80}?"
    r"([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
CONFIDENCE_MARKER_RE = re.compile(
    r"(?:<parameter=confidence>|\bconfidence\b|\bconfident\b)",
    re.IGNORECASE,
)


def has_contextgraph_finish_format(value: str) -> bool:
    """Return whether a response contains a valid ContextGraph finish call."""
    for match in FUNCTION_RE.finditer(str(value or "")):
        if match.group(1).strip().lower() != "finish":
            continue
        arguments = {
            key.strip().lower(): content.strip()
            for key, content in PARAMETER_RE.findall(match.group(2))
        }
        if arguments.get("answer"):
            return True
    return False


def _unwrap(value):
    if hasattr(value, "ndim") and value.ndim == 0:
        return value.item()
    if isinstance(value, (list, tuple)):
        return value[0]
    try:
        if getattr(value, "ndim", 0) > 0:
            return value[0]
    except (IndexError, TypeError):
        pass
    return value


def extract_numeric_answer(value: str) -> Decimal | None:
    """Extract a submitted answer, preferring explicit answer fields over prose."""
    text = str(value or "")
    candidate = None
    for pattern in ANSWER_BLOCK_RES:
        match = pattern.search(text)
        if match:
            numbers = NUMBER_RE.findall(match.group(1))
            if numbers:
                candidate = numbers[0]
                break
    if candidate is None:
        marker_match = ANSWER_MARKER_RE.search(text)
        if marker_match:
            candidate = marker_match.group(1)
    if candidate is None:
        # Emergency finalization may wrap an otherwise plain-text answer in the
        # finish tool. Ignore trailing confidence values before applying the
        # last-number fallback, otherwise "... 800 ... confidence 100%" is
        # incorrectly scored as 100.
        fallback_text = CONFIDENCE_MARKER_RE.split(text, maxsplit=1)[0]
        matches = NUMBER_RE.findall(fallback_text)
    else:
        matches = [candidate]
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def extract_fn_call(text: str):
    matches = list(FUNCTION_RE.finditer(text or ""))
    if not matches:
        return None
    match = matches[-1]
    return {
        "function": match.group(1).strip(),
        "arguments": {
            key.strip(): value.strip()
            for key, value in PARAMETER_RE.findall(match.group(2))
        },
    }


class MathEnv:
    """No-server GSM8K environment with deterministic exact numeric reward."""

    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.stats = Counter()
        self.judge_audit = []
        self.question = ""
        self.label_answer = ""
        self.predicted_answer = None
        self.final_response = None
        self.emergency_finish_wrapped = False
        self.is_finish = False
        self.env_fail = False

    async def init_env(self, item):
        extra = copy.deepcopy(_unwrap(item.non_tensor_batch["extra_info"]))
        self.question = str(extra["query"])
        self.label_answer = str(extra["answer"])
        self.predicted_answer = None
        self.final_response = None
        self.emergency_finish_wrapped = False
        self.is_finish = False
        self.judge_audit = []
        self.instance_info = extra
        self.instance_info["problem_statement"] = self.question

    async def run_action(self, response):
        self.stats["action"] += 1
        fn_call = extract_fn_call(response)
        if fn_call is None:
            return {"observation": "No function call was detected in the model response."}

        name = fn_call["function"]
        arguments = fn_call["arguments"]
        if name == "think":
            self.stats["think"] += 1
            reasoning = arguments.get("reasoning", "").strip()
            if not reasoning:
                return {"observation": '[Error] The "think" function requires reasoning.'}
            return {
                "observation": (
                    f"Intermediate calculation recorded: {reasoning}\n"
                    "Continue or submit the final answer."
                )
            }
        if name == "finish":
            answer = arguments.get("answer", "").strip()
            if not answer:
                return {"observation": '[Error] The "finish" function requires an answer.'}
            self.predicted_answer = answer
            self.final_response = str(response or "")
            self.is_finish = True
            self.stats["finish"] += 1
            self.stats["is_finish"] = 1
            return {"action": "finish"}
        return {"observation": f'[Error] The function "{name}" is not supported by MathEnv.'}

    async def score_answer(self, predicted_answer, audit_sink=None):
        expected = extract_numeric_answer(self.label_answer)
        predicted = extract_numeric_answer(predicted_answer)
        score = int(expected is not None and predicted is not None and expected == predicted)
        audit = {
            "question": self.question,
            "correct_answer": self.label_answer,
            "predicted_answer": str(predicted_answer),
            "strict_em": bool(score),
            "judge_model": None,
            "judge_method": "gsm8k_exact",
            "score": score,
        }
        if audit_sink is not None:
            audit_sink.append(copy.deepcopy(audit))
        print("[JUDGE AUDIT] " + json.dumps(audit, ensure_ascii=False))
        return score

    async def get_reward(self, item, messages, context):
        prediction = self.predicted_answer
        format_source = self.final_response
        if prediction is None:
            for message in reversed(messages or []):
                role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
                if role == "assistant":
                    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
                    prediction = str(content or "").strip()
                    format_source = prediction
                    break
        correctness = await self.score_answer(prediction or "", audit_sink=self.judge_audit)
        plugin = getattr(self.config, "plugin", None)

        def reward_weight(name, default):
            raw = plugin.get(name, default) if isinstance(plugin, Mapping) else getattr(plugin, name, default)
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
            return value

        correctness_weight = reward_weight("math_correctness_reward_weight", 1.0)
        format_weight = reward_weight("math_format_reward_weight", 0.0)
        format_valid = (
            not self.emergency_finish_wrapped
            and has_contextgraph_finish_format(format_source or "")
        )
        correctness_reward = correctness_weight * correctness
        format_reward = format_weight * float(format_valid)
        reward = correctness_reward + format_reward
        self.stats["math_correctness"] = int(correctness)
        self.stats["math_format_valid"] = int(format_valid)
        self.stats["math_correctness_reward"] = correctness_reward
        self.stats["math_format_reward"] = format_reward
        self.stats["math_total_reward"] = reward
        if self.judge_audit:
            self.judge_audit[-1].update({
                "format_valid": bool(format_valid),
                "correctness_reward": correctness_reward,
                "format_reward": format_reward,
                "total_reward": reward,
            })
        print("[MATH REWARD] " + json.dumps({
            "correctness": correctness,
            "format_valid": bool(format_valid),
            "correctness_reward": correctness_reward,
            "format_reward": format_reward,
            "total_reward": reward,
        }, ensure_ascii=False))
        self.stats["judge_calls"] = len(self.judge_audit)
        self.stats["judge_positive"] = sum(audit["score"] > 0 for audit in self.judge_audit)
        self.stats["judge_strict_em"] = sum(bool(audit["strict_em"]) for audit in self.judge_audit)
        self.stats["judge_llm"] = 0
        self.stats["judge_parse_failure"] = 0
        return "", reward, {}
