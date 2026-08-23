#!/usr/bin/env python3
"""Small API-based evaluator for ALFWorld and ScienceWorld trajectories."""

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
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.utils import CallAPI, TaskContext
from verl import DataProto


WORKFLOWS = {
    "alfworld": "react",
    "alfworld_branch": "fold",
    "alfworld_graph": "graph",
    "scienceworld": "react",
    "scienceworld_branch": "fold",
    "scienceworld_graph": "graph",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workflow", choices=sorted(WORKFLOWS), default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=16384)
    parser.add_argument("--response-length", type=int, default=16384)
    parser.add_argument("--max-turn", type=int, default=70)
    parser.add_argument("--consolidation-interval", type=int, default=8)
    parser.add_argument("--max-session", type=int, default=4)
    parser.add_argument("--branch-len", type=int, default=8192)
    parser.add_argument("--turn-max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--save-messages", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def process_item_for(workflow: str):
    kind = WORKFLOWS[workflow]
    if kind == "react":
        from agents.react_agent import process_item
    elif kind == "fold":
        from agents.fold_agent import process_item
    else:
        from agents.graph_agent_isolated import process_item
    return process_item


def make_config(args: argparse.Namespace, workflow: str):
    graph = WORKFLOWS[workflow] == "graph"
    return OmegaConf.create({
        "actor_rollout_ref": {"rollout": {
            "prompt_length": args.prompt_length,
            "response_length": args.response_length,
            "plugin": {
                "workflow": workflow,
                "max_turn": args.max_turn,
                "val_max_turn": args.max_turn,
                "max_session": args.max_session,
                "val_max_session": args.max_session,
                "session_timeout": 3600,
                "branch_len": args.branch_len,
                "turn_max_new_tokens": args.turn_max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "reasoning_effort": args.reasoning_effort,
                "val_response_length": args.response_length,
                "process_reward": "[flat,scope,graph]" if graph else "[flat,scope]",
                "max_traj": 4,
                "must_finish": False,
                "must_search": False,
                "double_check": False,
                "enable_summary": False,
                "enable_retrieval_memory": graph,
                "consolidation_interval": args.consolidation_interval if graph else 0,
                "lambda_compact": 0.1,
                "lambda_cost": 0.005,
                "scienceworld_max_steps": args.max_turn,
            },
        }},
    })


def row_workflow(row: dict[str, Any], override: str | None) -> str:
    workflow = override or (row.get("extra_info") or {}).get("workflow")
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unsupported interactive workflow: {workflow}")
    return workflow


def make_dataproto(row: dict[str, Any], workflow: str) -> DataProto:
    extra = copy.deepcopy(row.get("extra_info") or {})
    extra["workflow"] = workflow
    item = DataProto()
    item.non_tensor_batch = {
        "ability": np.array([row.get("ability", "")], dtype=object),
        "extra_info": np.array([extra], dtype=object),
        "uid": np.array([extra.get("task_id", "unknown")], dtype=object),
        "reward_model": np.array([row.get("reward_model", {})], dtype=object),
    }
    item.meta_info = {"generation_kwargs": {}, "max_turn": 0}
    return item


async def eval_one(row: dict[str, Any], args: argparse.Namespace, tokenizer) -> dict[str, Any]:
    workflow = row_workflow(row, args.workflow)
    item = make_dataproto(row, workflow)
    extra = item.non_tensor_batch["extra_info"][0]
    result = {
        "task_id": extra.get("task_id", "unknown"),
        "workflow": workflow,
        "status": "failed",
        "task_reward": 0.0,
        "agent_reward": 0.0,
        "is_finish": False,
    }
    client = None
    try:
        config = make_config(args, workflow)
        client = CallAPI(args.model_name, tokenizer, config.actor_rollout_ref.rollout)
        context = TaskContext(config, 0, False, tokenizer, client)
        output = await process_item_for(workflow)(item, context)
        out = output[0] if isinstance(output, list) else output
        fields = getattr(out, "extra_fields", {}) or {}
        stats = fields.get("env_stats", {}) or {}
        agent_reward = float(getattr(out, "reward_score", 0.0) or 0.0)
        task_reward = float(stats.get("task_reward", agent_reward) or 0.0)
        result.update({
            "status": "success",
            "task_reward": task_reward,
            "agent_reward": agent_reward,
            "is_finish": bool(fields.get("is_finish", False)),
            "termination_reason": fields.get("termination_reason", ""),
            "env_stats": stats,
        })
        if args.save_messages:
            result["messages"] = fields.get("messages", [])
            result["graph_trace"] = fields.get("graph_trace")
            result["graph_state"] = fields.get("graph_state", "")
            result["graph_rewards"] = fields.get("graph_rewards", {})
    except Exception as exc:
        result["error"] = repr(exc)
    finally:
        if client is not None:
            await client.close()
    return result


async def run(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, trust_remote_code=True, local_files_only=True
    )
    semaphore = asyncio.Semaphore(max(1, args.num_workers))

    async def guarded(row):
        async with semaphore:
            return await eval_one(row, args, tokenizer)

    return await asyncio.gather(*(guarded(row) for row in rows))


async def preflight(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        timeout=30.0,
    )
    try:
        response = await client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_completion_tokens=32,
        )
        if not response.choices:
            raise RuntimeError("Model API returned no choices")
    finally:
        await client.close()


async def run_with_preflight(
    rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Run API validation and evaluation on one event loop."""
    await preflight(args)
    return await run(rows, args)


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.data_path)
    frame = frame.iloc[args.start_index:]
    if args.max_samples > 0:
        frame = frame.head(args.max_samples)
    rows = frame.to_dict("records")
    if not rows:
        raise SystemExit(f"No rows selected from {args.data_path}")

    workflows = [row_workflow(row, args.workflow) for row in rows]
    if args.dry_run:
        for row, workflow in zip(rows, workflows):
            process_item_for(workflow)
            make_dataproto(row, workflow)
        print(json.dumps({"status": "dry_run_ok", "count": len(rows), "workflows": workflows}))
        return

    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
        raise SystemExit("OPENAI_API_KEY and OPENAI_BASE_URL are required")
    results = asyncio.run(run_with_preflight(rows, args))
    summary = {
        "data_path": args.data_path,
        "model_name": args.model_name,
        "count": len(results),
        "completed": sum(bool(result.get("is_finish")) for result in results),
        "task_successes": sum(float(result.get("task_reward", 0.0)) > 0 for result in results),
        "runner_successes": sum(result.get("status") == "success" for result in results),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"interactive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved to {path}")
    if summary["runner_successes"] != summary["count"]:
        raise SystemExit("One or more interactive trajectories failed before producing output")


if __name__ == "__main__":
    main()
