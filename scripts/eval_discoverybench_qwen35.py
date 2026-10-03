"""Matched DiscoveryBench evaluation through a local token-ID model endpoint.

Each task runs in its own process (the Python sandbox changes cwd). Generation
uses the existing code loops; only the separate HMS scorer determines results.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
MODEL = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
WORKFLOWS = {"contextgraph": "code_graph", "foldagent": "code_branch"}


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf8")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def task_key(identity):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", identity)[:100] + "-" + hashlib.sha256(identity.encode()).hexdigest()[:12]


def config_for(method, context_length):
    from omegaconf import OmegaConf
    if context_length not in (32768, 65536):
        raise ValueError("Expected 32768 or 65536 context")
    response = context_length - 8192
    return OmegaConf.create({"algorithm": {"adv_estimator": ""}, "actor_rollout_ref": {"rollout": {
        "prompt_length": 8192, "response_length": response, "plugin": {
            "workflow": WORKFLOWS[method], "max_turn": 100, "val_max_turn": 100,
            "max_session": 10, "val_max_session": 10, "session_timeout": 3600,
            "branch_len": response, "turn_max_new_tokens": 2048, "val_response_length": response,
            "process_reward": None, "max_traj": 11, "must_finish": False, "enable_summary": False,
            "sandbox_timeout": 60, "structured_graph_controller": method == "contextgraph",
            "controller_owned_tool_formatting": method == "contextgraph",
            "controller_action_policy": "balanced", "consolidation_interval": 5,
            "auto_prune_max_active": 12, "auto_bind_branch_edges": True,
            "lambda_compact": 0.0, "lambda_cost": 0.0,
            "apply_chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        }}}})


def load_tasks(data, samples):
    import pandas as pd
    from envs.discoverybench_env import load_metadata
    rows = pd.read_parquet(data).to_dict("records")
    tasks = []
    for row in rows:
        task = dict(row["extra_info"])
        if row["ability"] != "DiscoveryBench" or task.get("dataset_type") != "real" or task.get("dataset_split") != "test":
            raise ValueError("Expected DiscoveryBench real-test data")
        task["metadata"] = load_metadata(task["metadata"])
        task["input_files"] = list(task["input_files"])
        task["input_rel_paths"] = list(task["input_rel_paths"])
        if not task.get("gold_hypothesis") or not task["input_files"]:
            raise ValueError("Missing reference hypothesis or input files")
        tasks.append(task)
    tasks.sort(key=lambda t: t["task_id"])
    if len(tasks) != 239 or len({t["task_id"] for t in tasks}) != 239:
        raise ValueError("Expected exactly 239 unique real-test tasks before sampling")
    if samples != -1 and not 1 <= samples <= len(tasks):
        raise ValueError("samples must be -1 or 1..239")
    return tasks if samples == -1 else tasks[:samples]


def shard_tasks(tasks, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Expected shard-count >= 1 and 0 <= shard-index < shard-count")
    return tasks[index::count]


def summary(root, ids):
    records = [json.loads(p.read_text()) for i in ids if (p := root / "instances" / task_key(i) / "result.json").exists()]
    graded = [r for r in records if r["status"] == "graded"]
    errors = len(records) - len(graded)
    value = {"selected": len(ids), "completed": len(records), "pending": len(ids) - len(records),
             "graded": len(graded), "infrastructure_errors": errors,
             "valid_predictions": sum(r.get("valid_prediction", False) for r in graded),
             "mean_hms_graded": sum(r["hms"] for r in graded) / len(graded) if graded else None,
             "mean_hms": sum(r["hms"] for r in graded) / len(ids) if ids and len(graded) == len(ids) else None}
    save(root / "summary.json", value)
    print("DISCOVERY_PROGRESS " + json.dumps(value), flush=True)
    return value


def grade_prediction(task, workdir):
    from envs.discoverybench_env import load_prediction
    from envs.discoverybench_eval import score_hypothesis
    path = workdir / "pred_results/discovery_result.json"
    try:
        hypothesis, workflow = load_prediction(path)
    except (FileNotFoundError, ValueError, UnicodeError) as exc:
        return {"status": "graded", "hms": 0.0, "valid_prediction": False, "detail": str(exc)}
    # Judge exceptions propagate as infrastructure failures, never zero HMS.
    record = score_hypothesis(query=task["query"], gold_hypothesis=task["gold_hypothesis"],
                              gold_workflow=task.get("gold_workflow", ""), predicted_hypothesis=hypothesis,
                              predicted_workflow=workflow, metadata=task["metadata"], dataset_type="real")
    score = float(record["final_score"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid HMS score")
    return {"status": "graded", "hms": score, "valid_prediction": True,
            "prediction": {"hypothesis": hypothesis, "workflow": workflow}, "hms_audit": record}


async def generate(args, task, directory):
    import numpy as np
    from transformers import AutoTokenizer
    from verl import DataProto
    from agents.utils import TaskContext
    from scripts.eval_bcp_qwen38 import TokenClient, tokenizer_preflight
    if args.method == "contextgraph":
        from agents.graph_agent_code_isolated import process_item
    else:
        from agents.fold_agent_code import process_item
    config = config_for(args.method, args.context_length)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer_preflight(tokenizer, config.actor_rollout_ref.rollout)
    client = TokenClient(args.endpoint, MODEL, tokenizer, config.actor_rollout_ref.rollout,
                         42 + int(hashlib.sha256(task["task_id"].encode()).hexdigest()[:8], 16) % 1000000,
                         directory / "requests.jsonl")
    extra = copy.deepcopy(task)
    extra.update(workflow=WORKFLOWS[args.method], workdir=str(directory / "workdir"),
                 problem_statement=task["instruction"])
    item = DataProto()
    item.non_tensor_batch = {"ability": np.array(["DiscoveryBench"], dtype=object),
                             "extra_info": np.array([extra], dtype=object),
                             "uid": np.array([task["task_id"]], dtype=object),
                             "reward_model": np.array([{}], dtype=object)}
    item.meta_info = {"generation_kwargs": {}, "max_turn": 100}
    # Existing loops expect a reward call. Its format-only result is discarded;
    # no placeholder reward is exported as an evaluation score.
    os.environ["DISCOVERYBENCH_REAL_EVAL"] = "0"
    os.environ.pop("DISCOVERYBENCH_RESULTS_DIR", None)
    context = TaskContext(config=config, global_step=0, llm_client=client, is_train=False, tokenizer=tokenizer)
    try:
        output = await process_item(item, context)
        if not output or client.failed:
            raise RuntimeError("Empty rollout or failed model request")
        trajectories = []
        for rollout in output:
            fields = dict(rollout.extra_fields)
            fields["env_stats"] = {k: v for k, v in fields.get("env_stats", {}).items()
                                   if "reward" not in k and k not in ("get_final_score", "branch_success")}
            for k in ("graph_rewards", "judge_audit"):
                fields.pop(k, None)
            fields["grading_status"] = "pending_external_hms"
            trajectories.append(fields)
        save(directory / "trajectory.json", trajectories)
    finally:
        await client.client.aclose()


def preflight(args):
    import httpx
    from envs.discoverybench_eval import build_judge_client, _chat_json
    for name in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "xgboost", "matplotlib", "seaborn", "openpyxl"):
        __import__(name)
    tasks = load_tasks(args.data, args.samples)
    files = {str(Path(p).resolve()): digest(p) for t in tasks for p in t["input_files"]}
    judge, model, provider = build_judge_client()
    try:
        _chat_json(judge, model, 'Return a JSON object with the field "ok" equal to true.')
    finally:
        judge.close()
    if args.endpoint:
        with httpx.Client(trust_env=False, timeout=20) as client:
            response = client.get(args.endpoint.rstrip("/") + "/v1/models")
            response.raise_for_status()
            models = response.json()["data"]
        if not any(m["id"] == MODEL and m.get("max_model_len", 0) >= args.context_length for m in models):
            raise ValueError("Model identity or context limit mismatch")
    return tasks, files, {"model": model, "provider": provider}


def run(args):
    import fcntl
    from omegaconf import OmegaConf
    from scripts.run_swe_arm_subset import run_command
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks, files, judge = preflight(args)
        global_ids = [t["task_id"] for t in tasks]
        shard_index, shard_count = args.shard_index, args.shard_count
        tasks = shard_tasks(tasks, shard_index, shard_count)
        versions = {name: importlib.metadata.version(name) for name in (
            "transformers", "torch", "pandas", "numpy", "scipy", "scikit-learn", "statsmodels", "xgboost", "openai")}
        manifest = {"benchmark": "DiscoveryBench", "split": "real/test", "data_sha256": digest(args.data),
                    "inputs_sha256": files, "task_ids": [t["task_id"] for t in tasks], "method": args.method,
                    "global_task_ids": global_ids, "shard_index": shard_index, "shard_count": shard_count,
                    "model": MODEL, "model_revision": REVISION, "model_path": str(Path(args.model_path).resolve()),
                    "seed": 42, "judge": judge, "versions": versions, "task_timeout": args.task_timeout,
                    "config": OmegaConf.to_container(config_for(args.method, args.context_length)),
                    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
        path = root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Resume protocol mismatch: use the original code, data, environment and budget")
        save(path, manifest)
        ids = manifest["task_ids"]
        summary(root, ids)
        for index, task in enumerate(tasks):
            output = root / "instances" / task_key(task["task_id"])
            output.mkdir(parents=True, exist_ok=True)
            if (output / "result.json").exists():
                continue
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=output))
            save(attempt / "task.json", task)
            print(f"DISCOVERY_TASK {index + 1}/{len(tasks)} {task['task_id']} artifacts={attempt}", flush=True)
            started = time.monotonic()
            try:
                run_command([sys.executable, "-u", __file__, "--task", str(attempt),
                             "--method", args.method, "--model-path", args.model_path,
                             "--endpoint", args.endpoint, "--context-length", str(args.context_length)],
                            attempt / "generation.log", timeout=args.task_timeout)
                result = json.loads((attempt / "result.json").read_text())
            except Exception as exc:
                result = {"status": "infrastructure_error", "error": str(exc)}
            result.update(task_id=task["task_id"], attempt=str(attempt), elapsed_seconds=time.monotonic() - started)
            save(output / "result.json", result)
            summary(root, ids)
        final = summary(root, ids)
        return 2 if final["infrastructure_errors"] else 0


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt("Allocation terminating")
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=WORKFLOWS, default="contextgraph")
    parser.add_argument("--data", type=Path, default=REPO / "data/discoverybench_real_test_code.parquet")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--endpoint")
    parser.add_argument("--context-length", type=int, choices=(32768, 65536), default=65536)
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--task-timeout", type=int, default=7200)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        tasks, _, judge = preflight(args)
        print(f"DISCOVERY_PREFLIGHT_OK tasks={len(tasks)} judge={judge}")
        return
    if not args.model_path or not args.endpoint:
        parser.error("model-path and endpoint are required")
    if args.task:
        directory = args.task.resolve()
        task = json.loads((directory / "task.json").read_text())
        asyncio.run(generate(args, task, directory))
        save(directory / "result.json", grade_prediction(task, directory / "workdir"))
    else:
        if not args.output or args.task_timeout <= 0:
            parser.error("output and positive timeout are required")
        raise SystemExit(run(args))


if __name__ == "__main__":
    main()
