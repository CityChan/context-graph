"""Offline smoke preparation and strict artifact checks; never a benchmark score."""
import argparse
import json
import math
import os
from pathlib import Path
import re

from scripts.prepare_gram_data import prepare, sha256, write_json


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
    parser.add_argument("mode", choices=("prepare", "audit", "ray-ready"))
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
