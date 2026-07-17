#!/usr/bin/env python3
"""Evaluate GAIA parquets with ReAct, FoldAgent, or ContextGraph.

This API-based evaluator is intended for small validation subsets and smoke
runs. Distributed/vLLM training or large eval can use the generated GAIA
parquets directly with the existing verl entry points.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.utils import CallAPI, TaskContext
from verl import DataProto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GAIA search-agent parquets.")
    parser.add_argument("--data-path", default="data/gaia_validation_graph.parquet")
    parser.add_argument("--output-dir", default="results/gaia")
    parser.add_argument("--workflow", choices=["search", "search_branch", "search_graph"], default=None,
                        help="Override extra_info.workflow. Default reads workflow from parquet.")
    parser.add_argument("--model-name", default="gpt-5-nano")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--local-search-url", default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--prompt-length", type=int, default=16384)
    parser.add_argument("--response-length", type=int, default=32768)
    parser.add_argument("--max-turn", type=int, default=80)
    parser.add_argument("--max-session", type=int, default=8)
    parser.add_argument("--branch-len", type=int, default=8192)
    parser.add_argument("--turn-max-new-tokens", type=int, default=1024)
    parser.add_argument("--must-search", action="store_true",
                        help="Reject a correct final answer if the agent never searched.")
    parser.add_argument("--save-messages", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate parquet/workflow dispatch without calling a model API.")
    return parser.parse_args()


def _process_item_for_workflow(workflow: str):
    if workflow == "search":
        from agents.react_agent import process_item
    elif workflow == "search_branch":
        from agents.fold_agent import process_item
    elif workflow == "search_graph":
        from agents.graph_agent_isolated import process_item
    else:
        raise ValueError(f"Unsupported workflow: {workflow}")
    return process_item


def _make_config(args: argparse.Namespace, workflow: str):
    return OmegaConf.create({
        "actor_rollout_ref": {
            "rollout": {
                "prompt_length": args.prompt_length,
                "response_length": args.response_length,
                "plugin": {
                    "workflow": workflow,
                    "max_turn": args.max_turn,
                    "val_max_turn": args.max_turn,
                    "max_session": args.max_session,
                    "val_max_session": args.max_session,
                    "session_timeout": 5400,
                    "branch_len": args.branch_len,
                    "turn_max_new_tokens": args.turn_max_new_tokens,
                    "val_response_length": args.response_length,
                    "process_reward": "[flat,scope,graph]" if workflow == "search_graph" else "[flat,scope]",
                    "max_traj": 4,
                    "must_finish": False,
                    "double_check": False,
                    "must_search": bool(args.must_search),
                    "enable_summary": False,
                    "lambda_compact": 0.1,
                    "lambda_cost": 0.005,
                },
            }
        }
    })


def _row_workflow(row: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    extra = row.get("extra_info") or {}
    return extra.get("workflow") or "search_graph"


def _make_dataproto(row: dict[str, Any], workflow: str) -> DataProto:
    extra = copy.deepcopy(row.get("extra_info") or {})
    extra["workflow"] = workflow
    item = DataProto()
    item.non_tensor_batch = {
        "ability": np.array([row.get("ability", "GAIA")], dtype=object),
        "extra_info": np.array([extra], dtype=object),
        "uid": np.array([extra.get("instance_id") or extra.get("task_id") or "unknown"], dtype=object),
        "reward_model": np.array([row.get("reward_model", {})], dtype=object),
    }
    item.meta_info = {"generation_kwargs": {}, "max_turn": 0}
    return item


def _require_model_api(args: argparse.Namespace) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        return
    raise SystemExit(
        "OPENAI_API_KEY is required for eval_gaia.py because it uses the "
        "OpenAI-compatible API client. For an OpenAI-compatible local/proxy "
        "endpoint, set OPENAI_API_KEY=dummy and OPENAI_BASE_URL=<endpoint>/v1. "
        "Use --dry-run to validate data/workflow without model calls."
    )


def _score_from_output(output) -> tuple[float, dict[str, Any]]:
    out = output[0] if isinstance(output, list) else output
    if out is None:
        return 0.0, {}
    score = float(getattr(out, "reward_score", 0.0) or 0.0)
    extra = getattr(out, "extra_fields", {}) or {}
    return score, extra


async def eval_one(row: dict[str, Any], args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    workflow = _row_workflow(row, args.workflow)
    process_item = _process_item_for_workflow(workflow)
    config = _make_config(args, workflow)
    llm_client = CallAPI(url=args.model_name, tokenizer=tokenizer, config=config.actor_rollout_ref.rollout)
    context = TaskContext(config=config, global_step=0, llm_client=llm_client, is_train=False, tokenizer=tokenizer)
    item = _make_dataproto(row, workflow)
    extra = item.non_tensor_batch["extra_info"][0]
    result = {
        "task_id": extra.get("task_id") or extra.get("instance_id") or "unknown",
        "level": extra.get("level", ""),
        "workflow": workflow,
        "status": "failed",
        "score": 0.0,
    }
    try:
        output = await process_item(item, context)
        score, extra_fields = _score_from_output(output)
        result.update({
            "status": "success",
            "score": score,
            "env_stats": extra_fields.get("env_stats", {}),
        })
        if args.save_messages:
            result["messages"] = extra_fields.get("messages", [])
    except Exception as exc:
        result["error"] = repr(exc)
    return result


async def run_eval(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    semaphore = asyncio.Semaphore(max(1, args.num_workers))
    results: list[dict[str, Any]] = []

    async def guarded(row):
        async with semaphore:
            return await eval_one(row, args, tokenizer)

    tasks = [asyncio.create_task(guarded(row)) for row in rows]
    with tqdm(total=len(tasks), desc="GAIA", unit="item") as pbar:
        for fut in asyncio.as_completed(tasks):
            result = await fut
            results.append(result)
            scores = [r["score"] for r in results]
            pbar.set_postfix({"avg": f"{float(np.mean(scores)):.3f}", "id": result["task_id"]})
            pbar.update(1)
    return results


def main() -> None:
    args = parse_args()
    if args.local_search_url:
        os.environ["LOCAL_SEARCH_URL"] = args.local_search_url

    df = pd.read_parquet(args.data_path)
    if args.max_samples > 0:
        df = df.head(args.max_samples)
    rows = df.to_dict("records")
    if not rows:
        raise SystemExit(f"No rows in {args.data_path}")

    if args.dry_run:
        workflows = [_row_workflow(row, args.workflow) for row in rows]
        for row, workflow in zip(rows, workflows):
            _process_item_for_workflow(workflow)
            _make_dataproto(row, workflow)
        print(json.dumps({
            "status": "dry_run_ok",
            "data_path": args.data_path,
            "count": len(rows),
            "workflows": sorted(set(workflows)),
            "local_search_url": os.environ.get("LOCAL_SEARCH_URL", ""),
        }, indent=2))
        return

    _require_model_api(args)

    results = asyncio.run(run_eval(rows, args))
    scores = [r["score"] for r in results]
    by_level: dict[str, list[float]] = {}
    for r in results:
        by_level.setdefault(str(r.get("level", "")), []).append(float(r["score"]))
    summary = {
        "data_path": args.data_path,
        "workflow": args.workflow or "from_parquet",
        "model_name": args.model_name,
        "count": len(results),
        "avg_score": float(np.mean(scores)) if scores else 0.0,
        "by_level": {
            level: {"count": len(vals), "avg_score": float(np.mean(vals))}
            for level, vals in sorted(by_level.items())
        },
    }

    print(json.dumps(summary, indent=2))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"gaia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
