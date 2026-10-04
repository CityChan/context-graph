"""Pinned official WideSearch scoring in a dedicated process, with OpenAI transport.

The upstream metric functions, parsing and aggregation are unchanged. Transport
and malformed judge responses fail closed instead of becoming zero-score rows.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_agent_benchmarks import verify_evaluator
from scripts.eval_discoverybench_qwen35 import save


def validate_judge(content, prompt):
    from src.evaluation.metric_utils import parse_markdown_json
    value = parse_markdown_json(content)
    if not isinstance(value, dict):
        raise ValueError("WideSearch judge returned invalid JSON object")
    if prompt.startswith("You are an expert in grading answers."):
        body = prompt.split("====== response-start ======\n", 1)[1].split("\n====== response-end ======", 1)[0]
        keys = set(ast.literal_eval(body))
        if not keys.issubset(value):
            raise ValueError("WideSearch judge omitted row scores")
        if any(isinstance(value[k], bool) or not isinstance(value[k], (int, float))
               or not math.isfinite(value[k]) or value[k] not in (0, 1) for k in keys):
            raise ValueError("WideSearch judge produced invalid row scores")
    elif any(not isinstance(v, str) for v in value.values()):
        raise ValueError("WideSearch primary-key normalization must map strings to strings")
    return value


def grade(data, task, prediction, output, judge_model):
    from openai import OpenAI
    import pandas as pd
    data, output = Path(data), Path(output)
    upstream = data.parent / "official-evaluator"
    verify_evaluator(upstream)
    sys.path.insert(0, str(upstream))
    failures = []
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                    base_url=os.environ.get("WIDESEARCH_JUDGE_BASE_URL", "https://api.openai.com/v1"),
                    timeout=120, max_retries=2)

    def completion(messages, model_config_name=None, **kwargs):
        try:
            result = client.chat.completions.create(model=judge_model,
                         messages=[{"role": "user", "content": messages}], temperature=0,
                         max_completion_tokens=10240)
            if getattr(result.choices[0], "finish_reason", None) == "length":
                raise ValueError("WideSearch judge response exceeded its token budget")
            content = result.choices[0].message.content
            with (output.parent / "judge.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps({"prompt": messages, "response": content, "model": result.model,
                                         "usage": result.usage.model_dump() if result.usage else None}) + "\n")
            validate_judge(content, messages)
            return types.SimpleNamespace(content=content)
        except Exception as exc:
            failures.append(type(exc).__name__ + ": " + str(exc))
            raise

    # The upstream transport imports unrelated cloud SDKs and hardcodes Azure.
    # Supply the same callable contract without altering its evaluator source.
    transport = types.ModuleType("src.utils.llm")
    transport.llm_completion = completion
    sys.modules["src.utils.llm"] = transport
    try:
        from src.evaluation.data_loader import WideSearchQuery, WideSearchResponse
        from src.evaluation.evaluation import evaluate_single_query
        from src.utils.utils import norm_column
        references = json.loads((data.parent / "references.json").read_text(encoding="utf8"))
        ref = references[task["task_id"]]
        frame = pd.read_csv(io.StringIO(ref["csv"]))
        frame.columns = [norm_column(c) for c in frame.columns]
        frame = frame[ref["evaluation"]["required"]]
        query = WideSearchQuery(task["task_id"], task["query"], ref["evaluation"], frame, task["language"])
        answer = Path(prediction).read_text(encoding="utf8") if Path(prediction).exists() else ""
        response = WideSearchResponse(task["task_id"], answer)
        result = asdict(evaluate_single_query(query, response,
                        result_save_path=str(output.parent / "official-details.csv")))
        if failures or result["msg"].startswith("evaluator error"):
            raise RuntimeError("Official WideSearch evaluator failed: " + repr(failures or result["msg"]))
        for key, value in result.items():
            if key not in ("instance_id", "msg") and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid official metric {key}: {value}")
        save(output, {"status": "graded", "score": result["score"], "metrics": result,
                      "judge_model": judge_model, "valid_prediction": response.extract_dataframe() is not None,
                      "protocol": "official-metrics/custom-agent/tavily-live"})
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-model", default="gpt-4.1-2025-04-14")
    args = parser.parse_args()
    grade(args.data, json.loads(args.task.read_text(encoding="utf8")), args.prediction, args.output, args.judge_model)
