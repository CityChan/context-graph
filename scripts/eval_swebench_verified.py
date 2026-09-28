"""Prepare, generate, and officially grade SWE-bench Verified patches.

Generation uses existing code agent loops and a separate local vLLM endpoint.
The generation summary deliberately contains no accuracy/reward metric.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.swebench_env import PUBLIC_FIELDS, public_instance

DATASET = "princeton-nlp/SWE-bench_Verified"
HARNESS_COMMIT = "3f01bd622c0a22c00406139f69a234ef08225f22"  # v3.0.17
WORKFLOWS = {"react": "code", "foldagent": "code_branch", "contextgraph": "code_graph"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_tasks(rows, samples, seed, instance_ids=None):
    tasks = {row["instance_id"]: row for row in rows}
    if len(tasks) != len(rows):
        raise ValueError("Duplicate instance IDs")
    if instance_ids:
        if len(set(instance_ids)) != len(instance_ids) or set(instance_ids) - tasks.keys():
            raise ValueError("Unknown or duplicated requested instance IDs")
        return [tasks[key] for key in sorted(instance_ids)]
    keys = sorted(tasks)
    if samples != -1:
        if not 1 <= samples <= len(keys):
            raise ValueError("samples must be -1 or between 1 and dataset size")
        keys = sorted(random.Random(seed).sample(keys, samples))
    return [tasks[key] for key in keys]


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    # Resolve symbolic revisions once and use the immutable commit for download.
    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    rows = list(load_dataset(DATASET, split="test", revision=revision))
    if len(rows) != 500:
        raise ValueError(f"Expected 500 Verified test instances, got {len(rows)}")
    public = [public_instance(row) for row in rows]
    if len({row["instance_id"] for row in public}) != 500:
        raise ValueError("Duplicate Verified task IDs")
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "public").mkdir()
    (root / "grading").mkdir()
    write_json(root / "public/instances.json", public)
    write_json(root / "grading/instances.json", rows)
    manifest = {"dataset": DATASET, "revision": revision, "split": "test", "count": len(rows),
        "public_sha256": file_hash(root / "public/instances.json"),
        "grading_sha256": file_hash(root / "grading/instances.json"),
        "harness_commit": HARNESS_COMMIT}
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def load_public(root):
    manifest = read_json(root / "manifest.json")
    if manifest["dataset"] != DATASET or manifest["split"] != "test" or manifest["count"] != 500:
        raise ValueError("Not a prepared SWE-bench Verified test dataset")
    if file_hash(root / "public/instances.json") != manifest["public_sha256"]:
        raise ValueError("Public dataset hash mismatch")
    rows = read_json(root / "public/instances.json")
    if len(rows) != 500 or len({row["instance_id"] for row in rows}) != 500:
        raise ValueError("Incomplete or duplicated public dataset")
    if any(set(row) != set(PUBLIC_FIELDS) for row in rows):
        raise ValueError("Generation input must contain public fields only")
    return [public_instance(row) for row in rows], manifest


def config_for(args):
    from omegaconf import OmegaConf
    return OmegaConf.create({"algorithm": {"adv_estimator": ""}, "actor_rollout_ref": {"rollout": {
        "prompt_length": 8192, "response_length": 24576, "plugin": {
            "workflow": WORKFLOWS[args.method], "max_turn": args.max_turn,
            "max_session": 10, "val_max_session": 10, "session_timeout": args.task_timeout,
            "branch_len": 24576, "turn_max_new_tokens": 2048, "val_response_length": 24576,
            "process_reward": None, "max_traj": 11, "must_finish": False, "enable_summary": False,
            "structured_graph_controller": args.method == "contextgraph",
            "controller_action_policy": "balanced", "consolidation_interval": 5,
            "auto_prune_max_active": 12, "auto_bind_branch_edges": True,
            "lambda_compact": 0.0, "lambda_cost": 0.0,
            "swe_memory": args.memory, "swe_cpus": args.cpus, "swe_tool_timeout": 90,
            "apply_chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
        }}}})


def make_item(task, workflow):
    import numpy as np
    from verl import DataProto
    item = DataProto()
    item.non_tensor_batch = {
        "ability": np.array(["SWEVerified@" + json.dumps(public_instance(task))], dtype=object),
        "extra_info": np.array([{"instance_id": task["instance_id"], "workflow": workflow}], dtype=object),
        "uid": np.array([task["instance_id"]], dtype=object),
        "reward_model": np.array([{}], dtype=object),
    }
    item.meta_info = {"generation_kwargs": {}}
    return item


def non_scoring_stats(stats):
    # Existing loop reward fields are placeholders in generation-only mode.
    return {key: value for key, value in stats.items()
            if "reward" not in key and key not in ("get_final_score", "branch_success", "graph_shaping")}


async def generate_one(task, args, config, tokenizer, root, process_item):
    from agents.utils import TaskContext
    from envs.swebench_env import capture_environments, blocking_call
    from scripts.eval_bcp_qwen38 import TokenClient
    identity = task["instance_id"]
    output = root / "instances" / identity
    output.mkdir(parents=True, exist_ok=False)
    seed = args.seed + int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 1000000
    client = TokenClient(args.endpoint, args.model, tokenizer, config.actor_rollout_ref.rollout,
                         seed, output / "requests.jsonl")
    result = {"instance_id": identity, "status": "error", "grading_status": "pending"}
    prediction = {"instance_id": identity, "model_name_or_path": args.method + "--" + args.model,
                  "model_patch": ""}
    with capture_environments() as owned:
        try:
            # Agent loops mutate config in evaluation; isolate it per task.
            import copy
            local_config = copy.deepcopy(config)
            context = TaskContext(config=local_config, global_step=0, llm_client=client,
                                  is_train=False, tokenizer=tokenizer)
            item = make_item(task, WORKFLOWS[args.method])
            item.meta_info["max_turn"] = args.max_turn
            rollout = await asyncio.wait_for(process_item(item, context), timeout=args.task_timeout)
            if not rollout or client.failed or len(owned) != 1:
                raise RuntimeError("Missing rollout, model request failure, or unexpected environment count")
            env = owned[0]
            if env.env_fail or env.model_patch is None:
                raise RuntimeError("Container failed or patch extraction did not complete")
            extra = dict(rollout[0].extra_fields)
            extra["env_stats"] = non_scoring_stats(extra.get("env_stats", {}))
            extra["grading_status"] = "pending"
            write_json(output / "trajectory.json", extra)
            prediction["model_patch"] = env.model_patch
            result.update(status="generated", is_finish=bool(env.is_finish),
                patch_bytes=len(env.model_patch.encode()), env_stats=non_scoring_stats(env.stats),
                image=env.sandbox.provenance, seed=seed)
        except Exception as exc:
            result["error"] = repr(exc)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            for env in owned:
                try:
                    await blocking_call(env.close)
                except Exception as exc:
                    result.update(status="error", cleanup_error=repr(exc))
            await client.client.aclose()
    if result["status"] == "error":
        # Never silently submit a partial patch after an infrastructure failure.
        prediction["model_patch"] = ""
    (output / "model.patch").write_text(prediction["model_patch"], encoding="utf-8", newline="")
    write_json(output / "result.json", result)
    write_json(output / "prediction.json", prediction)
    print(json.dumps(result), flush=True)
    return result, prediction


async def generate(args):
    import docker
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from scripts.eval_bcp_qwen38 import TokenClient, tokenizer_preflight
    from agents.fold_agent_code import process_item as fold_process
    from agents.graph_agent_code_isolated import process_item as graph_process
    rows, dataset_manifest = load_public(Path(args.data_dir))
    selected = select_tasks(rows, args.samples, args.seed, args.instance_ids)
    client = docker.from_env(timeout=20)
    try:
        info = client.info()
        if info.get("OSType") != "linux" or info.get("Architecture") not in ("x86_64", "amd64"):
            raise RuntimeError("Use a Linux x86_64 Docker daemon for official Verified images")
    finally:
        client.close()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = config_for(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer_preflight(tokenizer, config.actor_rollout_ref.rollout)
    probe = TokenClient(args.endpoint, args.model, tokenizer, config.actor_rollout_ref.rollout,
                        args.seed, root / "preflight.jsonl")
    try:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": "Say OK."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        await probe.create_completion(ids, max_new_tokens=16)
        if args.method == "contextgraph":
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
                      "required": ["ok"], "additionalProperties": False}
            answer = await probe.create_completion(ids, max_new_tokens=64, structured_outputs={"json": schema})
            if not isinstance(json.loads(answer["choices"][0]["message"]["content"])["ok"], bool):
                raise RuntimeError("Model server failed structured output preflight")
    finally:
        await probe.client.aclose()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1], text=True, creationflags=creationflags).strip()
    write_json(root / "manifest.json", {"dataset": dataset_manifest, "method": args.method,
        "model": args.model, "model_path": str(Path(args.model_path).resolve()), "seed": args.seed,
        "instance_ids": [task["instance_id"] for task in selected], "commit": commit,
        "harness_commit": HARNESS_COMMIT, "config": OmegaConf.to_container(config),
        "agent_loop": "graph_agent_code_isolated" if args.method == "contextgraph" else "fold_agent_code",
        "protocol": "python_exec; network disabled; shared files across sequential branches; no gold hints",
        "tokenizer_files": {p.name: file_hash(p) for p in Path(args.model_path).iterdir()
                            if p.is_file() and p.suffix in (".json", ".jinja", ".txt")},
        "versions": {name: importlib.metadata.version(name) for name in ("transformers", "docker", "torch")}})
    semaphore = asyncio.Semaphore(args.workers)
    async def one(task):
        async with semaphore:
            result, pred = await generate_one(task, args, config, tokenizer, root,
                graph_process if args.method == "contextgraph" else fold_process)
            # Append immediately; interrupted runs retain completed work.
            for filename, value in (("results.jsonl", result), ("predictions.jsonl", pred)):
                with (root / filename).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(value) + "\n")
            return result
    results = await asyncio.gather(*(one(task) for task in selected))
    summary = {"method": args.method, "selected": len(selected),
        "generated": sum(row["status"] == "generated" for row in results),
        "generation_errors": sum(row["status"] == "error" for row in results),
        "empty_patches": sum(row.get("patch_bytes", 0) == 0 for row in results),
        "grading_status": "pending", "predictions_sha256": file_hash(root / "predictions.jsonl")}
    write_json(root / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["generation_errors"]:
        raise RuntimeError("Generation contains infrastructure errors; inspect instances/*/error.txt")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_predictions(root):
    manifest = read_json(root / "manifest.json")
    predictions = read_jsonl(root / "predictions.jsonl")
    results = read_jsonl(root / "results.jsonl")
    expected = sorted(manifest["instance_ids"])
    for rows in (predictions, results):
        if sorted(row["instance_id"] for row in rows) != expected:
            raise ValueError("Incomplete or duplicated generation rows; do not grade as a full run")
    summary = read_json(root / "generation_summary.json")
    if summary["predictions_sha256"] != file_hash(root / "predictions.jsonl"):
        raise ValueError("Predictions changed after generation")
    if any(row["status"] != "generated" for row in results):
        raise ValueError("Fix generation errors before official grading")
    if any(not isinstance(row.get("model_patch"), str) for row in predictions):
        raise ValueError("Invalid patch type")
    if len({row["model_name_or_path"] for row in predictions}) != 1:
        raise ValueError("Mixed model identifiers")
    return manifest, predictions


def grading_summary(root, grade_root):
    manifest, predictions = validate_predictions(root)
    results = read_jsonl(root / "results.jsonl")
    metadata = read_json(grade_root / "grading_manifest.json")
    if metadata["predictions_sha256"] != file_hash(root / "predictions.jsonl"):
        raise ValueError("Grading provenance mismatch")
    resolved, unresolved, empty, errors = [], [], [], []
    for pred in predictions:
        identity = pred["instance_id"]
        if not pred["model_patch"].strip():
            empty.append(identity)
            continue
        path = grade_root / "logs/run_evaluation/official" / pred["model_name_or_path"].replace("/", "__") / identity / "report.json"
        if not path.exists():
            errors.append(identity)
            continue
        report = read_json(path).get(identity, {})
        if not isinstance(report.get("resolved"), bool):
            errors.append(identity)
        elif report["resolved"]:
            resolved.append(identity)
        else:
            unresolved.append(identity)
    summary = {"benchmark": "SWE-bench_Verified", "method": manifest["method"],
        "count": len(predictions), "resolved": len(resolved), "unresolved": len(unresolved),
        "empty_patches": len(empty), "harness_errors": len(errors),
        "pass_at_1": len(resolved) / len(predictions) if not errors else None,
        "mean_tool_calls": sum(row.get("env_stats", {}).get("python_exec", 0) for row in results) / len(results),
        "finished": sum(bool(row.get("is_finish", False)) for row in results),
        "resolved_ids": sorted(resolved), "harness_error_ids": sorted(errors),
        "grading_complete": not errors, "harness_commit": HARNESS_COMMIT}
    write_json(grade_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if errors:
        raise RuntimeError("Official harness has missing/invalid reports; no clean Pass@1")
    return summary


def grade(args):
    import docker
    root = Path(args.output).resolve()
    manifest, predictions = validate_predictions(root)
    data_dir = Path(args.data_dir).resolve()
    _, dataset_manifest = load_public(data_dir)
    if manifest["dataset"] != dataset_manifest:
        raise ValueError("Generation and grading datasets differ")
    gold_path = data_dir / "grading/instances.json"
    if file_hash(gold_path) != dataset_manifest["grading_sha256"]:
        raise ValueError("Grading dataset hash mismatch")
    # Check the installed VCS pin, not just the public package version string.
    distribution = importlib.metadata.distribution("swebench")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != HARNESS_COMMIT:
        raise RuntimeError("Install requirements_swebench_eval.txt in the grading Python environment")
    client = docker.from_env(timeout=120)
    try:
        info = client.info()
        if info.get("OSType") != "linux" or info.get("Architecture") not in ("x86_64", "amd64"):
            raise RuntimeError("Grading requires a Linux x86_64 Docker daemon")
        for result in read_jsonl(root / "results.jsonl"):
            image = result["image"]
            # The upstream harness addresses :latest. Require the exact image
            # used for generation to still be present, so a moved tag cannot
            # silently change the test environment between phases.
            actual = client.images.get(image["image"])
            if actual.id != image["image_id"]:
                raise RuntimeError(f"Image changed since generation: {image['image']}")
    finally:
        client.close()
    grade_root = root / args.grade_dir
    grade_root.mkdir(parents=True, exist_ok=False)  # Never reuse cached reports for changed patches.
    identities = set(manifest["instance_ids"])
    selected = [row for row in read_json(gold_path) if row["instance_id"] in identities]
    if sorted(row["instance_id"] for row in selected) != sorted(identities):
        raise ValueError("Grading dataset is missing selected tasks")
    write_json(grade_root / "dataset.json", selected)
    write_json(grade_root / "grading_manifest.json", {"harness_commit": HARNESS_COMMIT,
        "predictions_sha256": file_hash(root / "predictions.jsonl"),
        "dataset_sha256": file_hash(grade_root / "dataset.json")})
    command = [sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", str(grade_root / "dataset.json"), "--split", "test",
        "--predictions_path", str(root / "predictions.jsonl"), "--run_id", "official",
        "--max_workers", str(args.workers), "--timeout", str(args.test_timeout),
        "--namespace", "swebench", "--cache_level", "instance"]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with (grade_root / "harness.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=grade_root, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=flags)
    if completed.returncode:
        raise RuntimeError(f"Official harness exited {completed.returncode}; see {grade_root / 'harness.log'}")
    grading_summary(root, grade_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Download and pin all 500 Verified test tasks")
    prep.add_argument("--data-dir", required=True)
    prep.add_argument("--revision", default="main")
    run = sub.add_parser("generate", help="Generate patches; no accuracy is computed")
    run.add_argument("--data-dir", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--method", choices=WORKFLOWS, required=True)
    run.add_argument("--model", default="Qwen/Qwen3.5-9B")
    run.add_argument("--model-path", required=True, help="Local tokenizer/checkpoint matching vLLM weights")
    run.add_argument("--endpoint", required=True, help="vLLM base URL without /v1; must return token IDs")
    run.add_argument("--samples", type=int, default=5)
    run.add_argument("--instance-ids", nargs="+")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--max-turn", type=int, default=100)
    run.add_argument("--task-timeout", type=int, default=3600)
    run.add_argument("--memory", default="8g")
    run.add_argument("--cpus", type=float, default=4)
    grading = sub.add_parser("grade", help="Run the pinned official Docker harness")
    grading.add_argument("--data-dir", required=True)
    grading.add_argument("--output", required=True)
    grading.add_argument("--grade-dir", default="grading")
    grading.add_argument("--workers", type=int, default=1)
    grading.add_argument("--test-timeout", type=int, default=1800)
    summary = sub.add_parser("summarize", help="Validate official reports without running containers")
    summary.add_argument("--output", required=True)
    summary.add_argument("--grade-dir", default="grading")
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1 or getattr(args, "cpus", 1) <= 0:
        parser.error("workers and cpus must be positive")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "generate":
        asyncio.run(generate(args))
    elif args.command == "grade":
        grade(args)
    else:
        grading_summary(Path(args.output), Path(args.output) / args.grade_dir)


if __name__ == "__main__":
    main()
