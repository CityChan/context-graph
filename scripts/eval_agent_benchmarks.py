"""Resumable paired-agent evaluation for ScienceWorld and DiscoveryWorld.

Each task has a process boundary, durable tool/request logs, a trajectory and an
atomic result. A model/JVM failure is an infrastructure error, not zero.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.eval_discoverybench_qwen35 import MODEL, REVISION, digest, save, task_key, shard_tasks
from scripts.prepare_agent_benchmarks import SCIENCEWORLD_VERSION
from scripts.generation_audit import combine_degeneration_stats, degeneration_stats, require_generation_quality

BENCHMARKS = ("scienceworld", "discoveryworld")
METHODS = ("contextgraph", "foldagent")
MEMORY_PROFILES = ("legacy", "turns", "repaired")
PROMPT_PROFILES = ("legacy", "focus_v2", "discoveryworld_v1")
DEFAULT_MEMORY_PROFILES = {"scienceworld": "turns", "discoveryworld": "repaired"}


def config_for(benchmark, method, context_length, max_steps=100, memory_profile=None, prompt_profile="legacy", observation_profile="full"):
    if benchmark not in BENCHMARKS:
        raise ValueError("Unknown benchmark")
    if observation_profile not in ("full", "compact_v1") or (benchmark != "discoveryworld" and observation_profile != "full"):
        raise ValueError("Compact observations are only supported for DiscoveryWorld")
    if memory_profile is None:
        memory_profile = DEFAULT_MEMORY_PROFILES[benchmark]
    if memory_profile not in MEMORY_PROFILES:
        raise ValueError("Unknown memory profile")
    if prompt_profile not in PROMPT_PROFILES:
        raise ValueError("Unknown prompt profile")
    if (benchmark == "discoveryworld") != (prompt_profile == "discoveryworld_v1"):
        raise ValueError("DiscoveryWorld requires discoveryworld_v1; ScienceWorld requires legacy or focus_v2")
    from scripts.eval_discoverybench_qwen35 import config_for as base_config
    config = base_config(method, context_length)
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.workflow = benchmark + ("_graph" if method == "contextgraph" else "_branch")
    if benchmark == "discoveryworld":
        plugin.discoveryworld_max_steps = max_steps
        plugin.max_turn = plugin.val_max_turn = max_steps
        plugin.discoveryworld_prompt_profile = prompt_profile
        plugin.discoveryworld_observation_profile = observation_profile
    else:
        plugin.scienceworld_max_steps = max_steps
    # Controller checkpoints consume tokens/time, but not task-turn opportunities.
    # Keep this in both manifests so the paired budget convention is explicit.
    plugin.graph_controller_counts_as_turn = memory_profile == "legacy"
    if benchmark == "scienceworld":
        plugin.scienceworld_memory_profile = memory_profile
        plugin.scienceworld_prompt_profile = prompt_profile
    if method == "contextgraph" and memory_profile == "repaired":
        plugin.contextgraph_memory_mode = "repaired"
        plugin.auto_prune_keep_recent = 8
        plugin.working_memory_keep_recent = 8
        plugin.retrieval_history_labels = True
        plugin.inject_graph_state_after_action = False
    plugin.final_answer_reserve = 0
    plugin.final_answer_safety_margin = 128
    plugin.turn_max_new_tokens = 2048
    return config


def load_tasks(path, benchmark, samples):
    if benchmark not in BENCHMARKS:
        raise ValueError("Unknown benchmark")
    bundle = json.loads(Path(path).read_text(encoding="utf8"))
    if bundle["source"]["benchmark"] != benchmark:
        raise ValueError("Benchmark/data mismatch")
    tasks = bundle["tasks"]
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Empty or duplicate task list")
    if benchmark == "discoveryworld":
        from envs.discoveryworld_protocol import REVISION, tasks_for
        source = bundle["source"]
        if source.get("revision") != REVISION or source.get("split") != "public" or source.get("seeds") != list(range(5)):
            raise ValueError("DiscoveryWorld public suite provenance mismatch")
        expected = tasks_for(source["difficulty"])
        if tasks != expected:
            raise ValueError("DiscoveryWorld suite must match the full pinned catalogue; select subsets with --samples")
        if samples != -1 and not 1 <= samples <= len(tasks):
            raise ValueError("Invalid sample count")
        return source, tasks if samples == -1 else tasks[:samples]
    for task in tasks:
        allowed = {"task_id", "task_name", "variation_idx", "split"}
        if set(task) != allowed:
            raise ValueError("Unexpected task fields; grading references must remain separate")
        if task["split"] != "test":
            raise ValueError("Formal evaluation requires the ScienceWorld test split")
    if samples != -1 and not 1 <= samples <= len(tasks):
        raise ValueError("samples must be -1 or between 1 and the dataset size")
    tasks = sorted(tasks, key=lambda t: t["task_id"])
    return bundle["source"], tasks if samples == -1 else tasks[:samples]


def summary(root, ids, benchmark):
    records = [json.loads(p.read_text(encoding="utf8")) for identity in ids
               if (p := root / "instances" / task_key(identity) / "result.json").exists()]
    graded = [r for r in records if r["status"] == "graded"]
    value = {"benchmark": benchmark, "selected": len(ids), "completed": len(records),
             "pending": len(ids) - len(records), "graded": len(graded),
             "infrastructure_errors": len(records) - len(graded),
             "mean_score_graded": sum(r["score"] for r in graded) / len(graded) if graded else None,
             "mean_score": sum(r["score"] for r in graded) / len(ids) if ids and len(graded) == len(ids) else None}
    value["score_scale"] = "0..100; negative terminal scores clipped to 0; raw scores retained per task"
    value["successes"] = sum(r["success"] for r in graded)
    if benchmark == "discoveryworld":
        value["score_scale"] = "official normalized procedural task score multiplied by 100"
        value["knowledge_score"] = None
        value["knowledge_evaluation"] = "not implemented; not a full three-metric DiscoveryWorld evaluation"
    value.update(combine_degeneration_stats(records))
    save(root / "summary.json", value)
    print("BENCHMARK_PROGRESS " + json.dumps(value), flush=True)
    require_generation_quality(value)
    return value


def run_command(command, log, timeout):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    with Path(log).open("w", encoding="utf8") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, **options)
        try:
            code = proc.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"Subprocess exit {code}; log={log}")
        finally:
            # Also reap orphaned simulator/JVM descendants after a parent exits.
            if os.name == "nt":
                if proc.poll() is None:
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW, check=False)
            else:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    time.sleep(0.2)
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            proc.wait()


async def generate(args, task, directory):
    import numpy as np
    from transformers import AutoTokenizer
    from verl import DataProto
    from agents.utils import TaskContext
    from scripts.eval_bcp_qwen38 import TokenClient, tokenizer_preflight
    if args.method == "contextgraph":
        from agents.graph_agent_isolated import process_item
    else:
        from agents.fold_agent import process_item
    config = config_for(args.benchmark, args.method, args.context_length, args.max_steps, args.memory_profile, args.prompt_profile, getattr(args, "discoveryworld_observation_profile", "full"))
    rollout_config = config.actor_rollout_ref.rollout
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer_preflight(tokenizer, rollout_config)
    seed = 42 + int(hashlib.sha256(task["task_id"].encode()).hexdigest()[:8], 16) % 1000000
    client = TokenClient(args.endpoint, MODEL, tokenizer, rollout_config, seed, directory / "requests.jsonl")
    extra = dict(task, workflow=rollout_config.plugin.workflow, tool_log=str(directory / "tools.jsonl"))
    if args.benchmark == "discoveryworld":
        extra.update(grading_log=str(directory / "scorecard.json"), problem_statement=task["scenario"])
    else:
        extra.update(simplification="", problem_statement=f"ScienceWorld {task['task_name']}")
    item = DataProto()
    ability = "DiscoveryWorld@real" if args.benchmark == "discoveryworld" else "ScienceWorld@real"
    item.non_tensor_batch = {"ability": np.array([ability], dtype=object), "extra_info": np.array([extra], dtype=object),
                             "uid": np.array([task["task_id"]], dtype=object), "reward_model": np.array([{}], dtype=object)}
    item.meta_info = {"generation_kwargs": {}, "max_turn": args.max_steps if args.benchmark == "discoveryworld" else 100}
    try:
        output = await process_item(item, TaskContext(config=config, global_step=0, llm_client=client,
                                                      is_train=False, tokenizer=tokenizer))
        if not output:
            raise RuntimeError("Empty agent rollout")
        trajectories = [dict(r.extra_fields) for r in output]
        save(directory / "trajectory.json", trajectories)
        stats = trajectories[0]["env_stats"]
        if client.failed or any(stats.get(k) for k in ("env_init_error", "env_error")):
            raise RuntimeError("Model or environment failure; see trajectory and request logs")
        score = float(stats["environment_score"])
        from envs.discoveryworld_protocol import PROTOCOL
        result_protocol = "scienceworld-1.2.3/test/unsimplified/shared-sequential-environment"
        if args.benchmark == "discoveryworld":
            result_protocol = PROTOCOL
            if getattr(args, "discoveryworld_observation_profile", "full") == "compact_v1":
                result_protocol += "/compact_v1"
        save(directory / "result.json", {"status": "graded", "score": max(0.0, min(100.0, score)),
               "raw_score": score, "success": bool(stats.get("completed")), "env_stats": stats,
               "termination_reason": trajectories[0].get("termination_reason"),
               "protocol": result_protocol})
    finally:
        await client.client.aclose()


def preflight(args):
    from omegaconf import OmegaConf
    import httpx
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit tracked changes before running a reproducible evaluation")
    source, tasks = load_tasks(args.data, args.benchmark, args.samples)
    protocol = {"source": source, "data_sha256": digest(args.data), "method": args.method,
                "model": MODEL, "model_revision": REVISION,
                "model_path": str(Path(args.model_path).resolve()), "seed": 42,
                "decoding": {"temperature": 0, "top_p": 1, "thinking": True},
                "server_execution": {"requested_enforce_eager": os.environ.get("SERVER_ENFORCE_EAGER", "1") == "1"},
                "config": OmegaConf.to_container(config_for(args.benchmark, args.method, args.context_length, args.max_steps, args.memory_profile, args.prompt_profile, getattr(args, "discoveryworld_observation_profile", "full"))),
                "task_timeout": args.task_timeout,
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "httpx", "pandas", "numpy", "omegaconf")},
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    if Path(args.model_path).name != REVISION:
        raise ValueError("Expected the pinned Qwen3.5-9B snapshot directory")
    with httpx.Client(timeout=20, trust_env=False) as client:
        response = client.get(args.endpoint.rstrip("/") + "/v1/models")
        response.raise_for_status()
        models = response.json()["data"]
    if not any(m["id"] == MODEL and m.get("max_model_len", 0) >= args.context_length
               and str(m.get("root", "")).rstrip("/").endswith(REVISION) for m in models):
        raise ValueError("Server model/checkpoint/context mismatch")
    if args.benchmark == "discoveryworld":
        from envs.discoveryworld_protocol import verify_install
        protocol["simulator"] = verify_install()
        protocol["observation"] = "official text UI only; no images; oracle scorecards grading-only"
        protocol["observation_profile"] = getattr(args, "discoveryworld_observation_profile", "full")
        protocol["knowledge_evaluation"] = "not implemented"
        protocol["versions"].update({p: importlib.metadata.version(p) for p in ("pygame", "pathfinding")})
        return tasks, protocol
    if importlib.metadata.version("scienceworld") != SCIENCEWORLD_VERSION or source.get("version") != SCIENCEWORLD_VERSION:
        raise ValueError("ScienceWorld version mismatch")
    java = subprocess.run(["java", "-version"], check=True, timeout=20, capture_output=True, text=True)
    protocol["simulator"] = SCIENCEWORLD_VERSION
    protocol["versions"]["py4j"] = importlib.metadata.version("py4j")
    protocol["java_version"] = (java.stdout + java.stderr).strip()
    return tasks, protocol


def run(args):
    from filelock import FileLock
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "run.lock"), timeout=0):
        tasks, protocol = preflight(args)
        tasks = shard_tasks(tasks, args.shard_index, args.shard_count)
        ids = [t["task_id"] for t in tasks]
        protocol.update(task_ids=ids, shard_index=args.shard_index, shard_count=args.shard_count)
        path = root / "manifest.json"
        if path.exists() and json.loads(path.read_text(encoding="utf8")) != protocol:
            raise ValueError("Resume protocol mismatch: retain code, data, model, dependencies and budgets")
        save(path, protocol)
        summary(root, ids, args.benchmark)
        for index, task in enumerate(tasks):
            directory = root / "instances" / task_key(task["task_id"])
            directory.mkdir(parents=True, exist_ok=True)
            result_path = directory / "result.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text(encoding="utf8"))
                if not args.retry_errors or previous["status"] != "infrastructure_error":
                    continue
            attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=directory))
            save(attempt / "task.json", task)
            started, stage = time.monotonic(), "generation"
            print(f"BENCHMARK_TASK {index+1}/{len(tasks)} {task['task_id']} artifacts={attempt}", flush=True)
            try:
                run_command([sys.executable, "-u", __file__, args.benchmark, "--task", str(attempt),
                             "--method", args.method, "--endpoint", args.endpoint, "--model-path", args.model_path,
                             "--context-length", str(args.context_length), "--max-steps", str(args.max_steps),
                             "--memory-profile", args.memory_profile,
                             "--prompt-profile", args.prompt_profile,
                             "--discoveryworld-observation-profile", getattr(args, "discoveryworld_observation_profile", "full")],
                            attempt / "generation.log", args.task_timeout)
                result = json.loads((attempt / "result.json").read_text(encoding="utf8"))
            except Exception as exc:
                result = {"status": "infrastructure_error", "stage": stage, "error": str(exc)}
            result.update(task_id=task["task_id"], attempt=str(attempt), elapsed_seconds=time.monotonic() - started)
            request_log = attempt / "requests.jsonl"
            result["generation_audit"] = degeneration_stats([request_log] if request_log.exists() else [])
            save(result_path, result)
            summary(root, ids, args.benchmark)
        final = summary(root, ids, args.benchmark)
        return 2 if final["infrastructure_errors"] else 0


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt("Allocation ending")
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("--method", choices=METHODS, default="contextgraph")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context-length", type=int, choices=[32768, 65536], default=65536)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--memory-profile", choices=MEMORY_PROFILES,
                        help="Default: turns for ScienceWorld, repaired for DiscoveryWorld")
    parser.add_argument("--prompt-profile", choices=PROMPT_PROFILES, default="legacy")
    parser.add_argument("--discoveryworld-observation-profile", choices=("full", "compact_v1"), default="full")
    parser.add_argument("--samples", type=int, default=-1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--task-timeout", type=int, default=3900)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--task", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.benchmark != "discoveryworld" and args.discoveryworld_observation_profile != "full":
        parser.error("--discoveryworld-observation-profile is only supported for DiscoveryWorld")
    if args.memory_profile is None:
        args.memory_profile = DEFAULT_MEMORY_PROFILES[args.benchmark]
    if args.max_steps < 1 or args.task_timeout < 1:
        parser.error("step and timeout limits must be positive")
    if args.task:
        asyncio.run(generate(args, json.loads((args.task / "task.json").read_text(encoding="utf8")), args.task))
        return 0
    if args.data is None or args.output is None:
        parser.error("--data and --output are required")
    args.data = args.data.resolve()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
