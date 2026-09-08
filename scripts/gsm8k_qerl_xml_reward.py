"""QeRL GSM8K XML rewards exposed through verl's custom reward API."""

from __future__ import annotations

import re


SOFT_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, extra_info, kwargs
    extracted = extract_xml_answer(solution_str)
    correctness_reward = 2.0 if extracted == str(ground_truth) else 0.0
    soft_format_reward = 0.2 if SOFT_FORMAT_RE.search(solution_str) else 0.0
    return {
        "score": correctness_reward + soft_format_reward,
        "correctness": float(correctness_reward > 0.0),
        "correctness_reward": correctness_reward,
        "soft_format_valid": float(soft_format_reward > 0.0),
        "soft_format_reward": soft_format_reward,
    }
