"""Audited GRAM inference against explicit actor and memory-model endpoints."""
import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path

from agents.gram_agent import GramConfig, MemoryBackend, make_policy, run_episode, score_episode
from scripts.prepare_gram_data import sha256, write_json


def jsonl_writer(path):
    def write(event):
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return write


def summarize(results, selected):
    graded = [r for r in results if r["status"] == "graded"]
    if results and any("benchmark" in r and r["benchmark"] == "bcp" for r in results):
        return {"benchmark": "bcp", "selected": selected, "completed": len(results),
                "pending": selected-len(results), "graded": len(graded),
                "infrastructure_errors": len(results)-len(graded),
                "resolved": sum(r["score"] for r in graded),
                "answered": sum(r.get("termination_reason") == "answer" for r in graded),
                "blank_predictions": sum(not r.get("prediction", "").strip() for r in graded),
                "documents_consumed": sum(r.get("documents_consumed", 0) for r in graded),
                "no_change_memory_edits": sum(r.get("no_change_memory_edits", 0) for r in graded),
                "duplicate_retrievals_blocked": sum(r.get("duplicate_retrievals_blocked", 0) for r in graded),
                "mean_format_reward_graded": sum(r.get("format_reward", 0) for r in graded)/len(graded) if graded else None,
                "accuracy_graded": sum(r["score"] for r in graded)/len(graded) if graded else None,
                "accuracy": sum(r["score"] for r in graded)/selected if len(graded) == selected else None}
    return {"selected": selected, "completed": len(results), "pending": selected-len(results),
            "graded": len(graded), "infrastructure_errors": len(results)-len(graded),
            "mean_answer_f1_graded": sum(r["answer_f1"] for r in graded)/len(graded) if graded else None,
            "mean_answer_f1": sum(r["answer_f1"] for r in graded)/selected if len(graded) == selected else None,
            "answered": sum(r["termination_reason"] == "answer" for r in graded)}


async def evaluate(args):
    import httpx
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from scripts.eval_bcp_qwen38 import TokenClient

    bcp = getattr(args, "benchmark", "document-stream") == "bcp"
    if bcp:
        from agents.gram_bcp import BcpRetrieval, RETRIEVAL_LIMITS, load_tasks, score_bcp
        if os.getenv("OPENAI_API_KEY", "") in {"", "dummy"}:
            raise ValueError("BC-P requires a real OPENAI_API_KEY for the existing answer judge")
        if not os.getenv("LOCAL_SEARCH_URL"):
            raise ValueError("BC-P requires LOCAL_SEARCH_URL")
        tasks, refs, indices = load_tasks(args.data, args.samples, args.seed, args.shard_index, args.shard_count)
        data_manifest = {"benchmark": "bcp", "parquet_sha256": sha256(args.data), "indices": indices,
                         "retrieval": "local_bcp_corpus", "limits": RETRIEVAL_LIMITS,
                         "judge_model": os.getenv("JUDGE_MODEL", "gpt-5-nano")}
    else:
        tasks_path, refs_path = args.data / "tasks.json", args.data / "references.json"
        data_manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in data_manifest["files"].items():
            if sha256(args.data / name) != digest:
                raise ValueError(f"Prepared data changed: {name}")
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
        refs = json.loads(refs_path.read_text(encoding="utf-8"))
        if args.samples > 0:
            tasks = tasks[:args.samples]
        tasks = tasks[args.shard_index::args.shard_count]
    if not tasks:
        raise ValueError("Empty evaluation shard")
    config = GramConfig(max_steps=args.max_steps, max_episode_tokens=args.episode_tokens,
                        max_step_tokens=args.step_tokens, timeout_seconds=args.timeout,
                        search_hops=args.search_hops, search_top_k=args.search_top_k,
                        entity_threshold=args.entity_threshold,
                        action_decoding=getattr(args, "action_decoding", "unconstrained"),
                        bcp_progress_limit=getattr(args, "bcp_progress_limit", 0))
    repository = Path(__file__).resolve().parents[1]
    sources = ["agents/gram_agent.py", "agents/gram_memory.py", "agents/gram_prompts.py",
               "agents/utils.py", "scripts/eval_gram.py", "scripts/eval_bcp_qwen38.py",
               "scripts/prepare_gram_data.py", "agents/gram_decoding.py", "agents/structured_outputs.py"]
    if bcp:
        sources += ["agents/gram_bcp.py", "envs/local_search.py", "envs/judge_client.py"]
    manifest = {"protocol": "gram-bcp-adaptation-v1" if bcp else "gram-document-stream-v1", "paper_exact_reproduction": False,
                "data": data_manifest, "config": asdict(config), "task_ids": [t["task_id"] for t in tasks],
                "actor": {"model": args.model, "declared_revision": args.model_revision},
                "memory": {"model": args.memory_model, "declared_revision": args.memory_revision,
                           "output_protocol": "gram-memory-json-schema-v2-entity-enum"},
                "entity_matching": "cosine" if args.embedding_model else "normalized-exact-plus-helper-canonicalization",
                "embedding_model": args.embedding_model, "embedding_revision": args.embedding_revision,
                "context_length": args.context_length, "seed": args.seed, "temperature": 0.0,
                "template_enable_thinking": False,
                "source_sha256": {p: sha256(repository / p) for p in sources}}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    # Hash tokenizer artifacts, not the model weights. Revision strings are declarations, not attestation.
    manifest["tokenizer_sha256"] = {p.name: sha256(p) for p in sorted(Path(args.model_path).glob("*"))
                                    if p.is_file() and ("token" in p.name or "template" in p.name)}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume protocol mismatch: use a new output directory")
    else:
        if any(p.name != ".run.lock" for p in args.output.iterdir()):
            raise ValueError("Nonempty run directory has no compatible manifest")
        write_json(manifest_path, manifest)
    async with httpx.AsyncClient(timeout=30, trust_env=False) as probe:
        observed = {}
        for label, endpoint, model in (("actor", args.endpoint, args.model),
                                        ("memory", args.memory_endpoint, args.memory_model)):
            response = await probe.get(endpoint.rstrip("/") + "/v1/models")
            response.raise_for_status()
            models = response.json()["data"]
            matches = [m for m in models if m["id"] == model]
            if not matches:
                raise ValueError(f"{label} model is not served: {model}")
            if label == "actor" and matches[0].get("max_model_len", args.context_length) < args.context_length:
                raise ValueError("Actor server context is smaller than the requested protocol")
            observed[label] = matches[0]
        write_json(args.output / "observed_servers.json", observed)
    rollout = OmegaConf.create({"prompt_length": args.context_length-args.step_tokens-32,
                               "response_length": args.step_tokens+32,
                               "plugin": {"enable_summary": False, "retry_cjk": 0,
                                          "turn_max_new_tokens": args.step_tokens}})
    results = []
    from hashlib import sha256 as hash_bytes
    for task in tasks:
        key = hash_bytes(task["task_id"].encode()).hexdigest()[:20]
        directory = args.output / "instances" / key
        path = directory / "result.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["status"] == "graded" or not args.retry_errors:
                results.append(saved)
                continue
        directory.mkdir(parents=True, exist_ok=True)
        import tempfile
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=directory))
        audit = jsonl_writer(attempt / "trajectory.jsonl")
        client = TokenClient(args.endpoint, args.model, tokenizer, rollout,
                             args.seed + int(task["task_id"].split("-")[-1]) if bcp else args.seed,
                             attempt / "actor_requests.jsonl")
        memory = MemoryBackend(args.memory_endpoint, args.memory_model, audit=audit,
                               embedding_endpoint=args.embedding_endpoint, embedding_model=args.embedding_model)
        print(f"GRAM_TASK {len(results)+1}/{len(tasks)} {task['task_id']} artifacts={attempt}", flush=True)
        retrieval = None
        try:
            retrieval = BcpRetrieval() if bcp else None
            result = await run_episode(task, make_policy(client, tokenizer, rollout), memory, config, audit, retrieval=retrieval)
            write_json(attempt / "segments.json", result.pop("segments"))
            write_json(attempt / "episode.json", result)
            if bcp:
                result.update(await score_bcp(task["question"], refs[task["task_id"]], result["prediction"]))
                result["retrieval_stats"] = dict(retrieval.env.stats)
            else:
                result.update(score_episode(result, refs[task["task_id"]], config.process_weight))
            result["status"] = "graded"
        except Exception as exc:
            result = {"task_id": task["task_id"], "status": "infrastructure_error",
                      "error": f"{type(exc).__name__}: {exc}"}
        finally:
            await asyncio.gather(client.client.aclose(), memory.aclose(),
                                 *([retrieval.aclose()] if retrieval is not None else []))
        if bcp:
            result["benchmark"] = "bcp"
        result["attempt"] = str(attempt.resolve())
        write_json(path, result)
        results.append(result)
        summary = summarize(results, len(tasks))
        write_json(args.output / "summary.json", summary)
        print("GRAM_PROGRESS " + json.dumps(summary), flush=True)
    summary = summarize(results, len(tasks))
    write_json(args.output / "summary.json", summary)
    return 2 if summary["infrastructure_errors"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["document-stream", "bcp"], default="document-stream")
    for key in ("data", "output", "model-path"):
        parser.add_argument("--"+key, required=True, type=Path)
    for key in ("endpoint", "model", "model-revision", "memory-endpoint", "memory-model", "memory-revision"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--embedding-endpoint")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--entity-threshold", type=float, default=0.9)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--episode-tokens", type=int, default=32768)
    parser.add_argument("--step-tokens", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--action-decoding", choices=["unconstrained", "xml_regex"], default="unconstrained")
    parser.add_argument("--bcp-progress-limit", type=int, default=0,
                        help="Enable BC-P progress guard; maximum memory searches between retrieval/document consumption (0 disables)")
    parser.add_argument("--search-hops", type=int, default=2)
    parser.add_argument("--search-top-k", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.samples == 0 or args.samples < -1:
        parser.error("Invalid sample count or shard")
    if args.context_length <= args.step_tokens+32:
        parser.error("Context must accommodate prompt and step budget")
    if bool(args.embedding_model) != bool(args.embedding_endpoint) or (args.embedding_model and not args.embedding_revision):
        parser.error("Embedding model, endpoint and revision must be specified together")
    args.output.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        raise SystemExit(asyncio.run(evaluate(args)))


if __name__ == "__main__":
    main()
