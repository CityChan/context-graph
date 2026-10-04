"""Smoke preparation, GPU sampler preflight and artifact checks; not a benchmark score."""
import argparse
import json
import math
import os
from pathlib import Path
import re

from scripts.prepare_gram_data import prepare, sha256, write_json


async def check_memory_schema(root, endpoint, model):
    """Probe helper grammars and a simple positive extraction before loading the RL model."""
    from agents.gram_agent import MemoryBackend
    from agents.gram_prompts import ENTITIES, RELATIONS, MAINTENANCE
    from scripts.eval_gram import jsonl_writer
    root = Path(root)
    helper = MemoryBackend(endpoint, model, max_tokens=256,
                           audit=jsonl_writer(root / "memory-schema-preflight.jsonl"))
    payload = {"question": "Where was Alice born?", "requested_facts": "Alice was born in Paris.",
               "document": {"id": "schema-probe", "title": "Synthetic probe", "text": "Alice was born in Paris."},
               "existing_entities": [], "entities": ["Alice", "Paris"], "graph": []}
    checked = []
    try:
        for operation, prompt in (("entities", ENTITIES), ("relations", RELATIONS), ("maintenance", MAINTENANCE)):
            value = await helper.json_call(operation, prompt, payload)
            if operation == "entities" and not {"Alice", "Paris"}.issubset(value):
                raise ValueError("Helper semantic preflight failed: explicit Alice/Paris entities were not extracted")
            if operation == "relations" and not any(row[0] == "Alice" and row[2] == "Paris" for row in value):
                raise ValueError("Helper semantic preflight failed: explicit Alice-to-Paris fact was not extracted")
            if operation == "relations" and any(row[0] not in payload["entities"] or row[2] not in payload["entities"] for row in value):
                raise ValueError("Helper semantic preflight failed: relation endpoint escaped its entity enum")
            checked.append(operation)
    finally:
        await helper.aclose()
    report = {"output_protocol": "gram-memory-json-schema-v2-entity-enum", "checked": checked,
              "model": model, "purpose": "synthetic format and nonempty extraction probe; not benchmark accuracy"}
    write_json(root / "memory-schema-preflight.json", report)
    print("GRAM_MEMORY_SCHEMA_OK " + json.dumps(report), flush=True)


def exercise_native_sampler(sampler, torch, device="cuda"):
    """Exercise the selected dispatch, not a directly called fallback method."""
    backend = getattr(sampler.forward, "__name__", "")
    if backend != "forward_native":
        raise RuntimeError(f"Expected native vLLM sampler, selected {backend!r}")
    logits = torch.randn(2, 128, device=device, dtype=torch.float32)
    k = torch.tensor([10, 20], device=device, dtype=torch.int32)
    p = torch.tensor([0.9, 0.95], device=device, dtype=torch.float32)
    tokens, _ = sampler(logits, generators={}, k=k, p=p)
    if tokens.numel() != 2 or not ((tokens >= 0) & (tokens < 128)).all().item():
        raise RuntimeError("Native sampler returned invalid token IDs")
    return {"backend": backend, "sampled_tokens": tokens.numel()}


def check_native_sampler(root):
    # These imports belong after training_env has set the switch, in a fresh
    # process on each trainer node. This probe does not load model weights.
    import socket
    import torch
    import vllm
    import vllm.envs as vllm_envs
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0" or vllm_envs.VLLM_USE_FLASHINFER_SAMPLER:
        raise RuntimeError("vLLM did not honor VLLM_USE_FLASHINFER_SAMPLER=0")
    if not torch.cuda.is_available():
        raise RuntimeError("Sampler preflight requires a trainer GPU")
    report = exercise_native_sampler(TopKTopPSampler(logprobs_mode="raw_logprobs"), torch)
    torch.cuda.synchronize()
    report.update(host=socket.gethostname(), vllm_version=vllm.__version__,
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  VLLM_USE_FLASHINFER_SAMPLER="0")
    write_json(Path(root) / f"sampler-{report['host']}.json", report)
    print("GRAM_NATIVE_SAMPLER_OK " + json.dumps(report), flush=True)


def prepare_smoke(root, source):
    root, source = Path(root), Path(source)
    rows = json.loads(source.read_text(encoding="utf-8"))
    provenance = json.loads(source.with_name("source.json").read_text(encoding="utf-8"))
    if sha256(source) != provenance["fixture_sha256"] or len(rows) != 2:
        raise ValueError("Bundled HotpotQA fixture hash/count mismatch")
    for index, role in enumerate(("train", "validation")):
        raw = root / f"{role}-source.json"
        if raw.exists() or (root / role).exists():
            raise ValueError("Smoke data must use a fresh output directory")
        write_json(raw, [rows[index]])
        prepare(raw, root / role, "hotpotqa", role, parquet=True)
    from scripts.train_gram import check_data
    check_data(root / "train/data.parquet", root / "validation/data.parquet")
    write_json(root / "smoke-data.json", {
        "purpose": "optimizer mechanics only; not benchmark training or evaluation",
        "upstream_split": "HotpotQA distractor validation",
        "optimization_row": 0, "validation_row": 1,
        "source_sha256": sha256(source), "source_task_ids": [r["_id"] for r in rows],
        "all_documents_preserved": True, "held_out_benchmark_claim": False,
    })


def audit(root, world_size=3, steps=2, benchmark="document-stream"):
    root = Path(root)
    checkpoint = root / "checkpoints"
    if (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() != str(steps):
        raise ValueError("Final checkpoint marker missing or wrong")
    for rank in range(world_size):
        for kind in ("model", "optim", "extra_state"):
            path = checkpoint / f"global_step_{steps}/actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing or empty checkpoint shard: {path}")
    log = re.sub(r"\x1b\[[0-9;]*m", "", (root / "trainer.log").read_text(errors="replace"))
    metrics = {}
    for line in log.splitlines():
        match = re.search(r"\bstep:(\d+)\s+-\s+", line)
        if not match:
            continue
        values = dict(re.findall(r"([\w/@.+-]+):(-?(?:\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|inf|nan))(?=\s|$)", line))
        if "actor/grad_norm" in values:
            metrics[int(match[1])] = {key: float(value) for key, value in values.items()}
    for step in range(1, steps + 1):
        values = metrics.get(step, {})
        if not {"actor/grad_norm", "actor/kl_loss"} <= values.keys():
            raise ValueError(f"Missing optimizer/KL metrics at step {step}")
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError(f"Nonfinite training metrics at step {step}")
    if not any(values["actor/grad_norm"] > 0 for values in metrics.values()):
        raise ValueError("All gradient norms are zero; usable update signal not demonstrated")
    counts = {"memory_call": 0, "action": 0}
    bcp_rewards = []
    for path in (root / "traces").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("kind") in counts:
                counts[event["kind"]] += 1
            if event.get("kind") == "bcp_reward":
                bcp_rewards.append(event)
    if not all(counts.values()):
        raise ValueError("Missing actual actor/memory-helper trace events")
    if benchmark == "bcp":
        for validation in (False, True):
            rows = [r for r in bcp_rewards if r["validation"] == validation]
            if not rows or not any(r["external_searches"] > 0 for r in rows):
                raise ValueError("Missing BC-P training/validation retrieval and grading evidence")
            if any(r["task_reward"] not in (0, 1) or not r["judge_audit"] for r in rows):
                raise ValueError("Invalid BC-P reward audit")
    report = {"steps": steps, "world_size": world_size, "trace_events": counts,
              "optimizer_metrics": metrics, "checkpoint_reload_verified": False,
              "benchmark_performance_verified": False, "benchmark": benchmark,
              "bcp_graded_episodes": len(bcp_rewards)}
    write_json(root / "smoke-audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit", "ray-ready", "sampler-check", "memory-check"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--benchmark", choices=("document-stream", "bcp"), default="document-stream")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.benchmark == "bcp":
            from scripts.prepare_gram_bcp_rl import prepare_bcp
            prepare_bcp(args.run / "data", os.environ["GRAM_BCP_TRAIN_SOURCE"], os.environ["GRAM_BCP_VAL_SOURCE"])
        else:
            prepare_smoke(args.run / "data", Path("examples/gram/hotpotqa_dev_first2.json"))
    elif args.mode == "sampler-check":
        check_native_sampler(args.run)
    elif args.mode == "memory-check":
        import asyncio
        asyncio.run(check_memory_schema(args.run, os.environ["GRAM_MEMORY_ENDPOINT"], os.environ["GRAM_MEMORY_MODEL"]))
    elif args.mode == "audit":
        print(json.dumps(audit(args.run, world_size=args.world_size, benchmark=args.benchmark)))
    else:
        import ray
        ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
        try:
            nodes = [n for n in ray.nodes() if n["Alive"]]
            if len(nodes) != args.world_size or sum(n["Resources"].get("GPU", 0) for n in nodes) != args.world_size:
                raise SystemExit(f"Waiting for exactly {args.world_size} trainer nodes/GPUs")
            print(f"GRAM_RAY_READY nodes={args.world_size} GPUs={args.world_size}")
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
