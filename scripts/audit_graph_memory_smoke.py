"""Audit a completed three-shard BC-P memory smoke, without a success-score gate."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_records import read_evaluation
from scripts.generation_audit import require_generation_quality, degeneration_stats


def task_diagnostics(idx, trajectory):
    records = trajectory.get("model_contexts", [])
    errors = [r for r in records if r.get("format_error")]
    memory_errors = [r for r in trajectory.get("memory_audit", []) if r.get("error")]
    stats = trajectory.get("env_stats", {})
    def memory_detail(record):
        index = record.get("request_index")
        request = records[index] if type(index) is int and 0 <= index < len(records) else {}
        return dict(phase=record.get("phase"), error=record["error"], request_index=index,
                    output_tokens=len(request.get("output_ids", [])), max_tokens=request.get("max_tokens"),
                    response_tail=request.get("response", "")[-1600:])
    return dict(source_index=idx, stop=trajectory.get("termination_reason"),
                stats={k: stats.get(k) for k in ("environment_steps", "main_turn", "main_len",
                    "main_context_tokens", "actor_requests", "memory_requests", "invalid_tool",
                    "invalid_memory", "memory_updates", "hit_token_limit", "hit_max_turn", "hit_timeout")},
                format_error_events=len(errors),
                last_format_errors=[dict(phase=r.get("phase"), error=r["format_error"],
                    output_tokens=len(r.get("output_ids", [])), max_tokens=r.get("max_tokens"),
                    at_output_limit=bool(r.get("max_tokens") and len(r.get("output_ids", [])) >= r["max_tokens"]),
                    response_tail=r.get("response", "")[-1600:]) for r in errors[-3:]],
                memory_error_events=len(memory_errors),
                last_memory_errors=[memory_detail(r) for r in memory_errors[-3:]])


def audit(root):
    root = Path(root)
    manifests, results, summary = read_evaluation(root)
    method = manifests[0]["method"]
    if method not in {"memobrain", "amem"}:
        raise ValueError("Expected memobrain or amem")
    if summary["execution_errors"] or summary["judge_parse_failures"]:
        raise ValueError("Execution or judge failures in smoke")
    counts = dict(memory_requests=0, memory_updates=0, memory_recalls=0, invalid_memory=0,
                  environment_steps=0, generated_tokens=0, total_token=0, embedding_calls=0,
                  memory_nodes=0, memory_edges=0, memobrain_folds=0, memobrain_flushes=0, amem_evolutions=0)
    tasks = []
    for row in results:
        idx = row["source_index"]
        trajectory = json.loads((root / f"trajectory-{idx}.json").read_text(encoding="utf-8"))
        stats = trajectory["env_stats"]
        tasks.append(task_diagnostics(idx, trajectory))
        requests = [json.loads(s) for s in (root / f"requests-{idx}.jsonl").read_text().splitlines() if s.strip()]
        records = trajectory["model_contexts"]
        if len(records) != len(requests):
            raise ValueError(f"Task {idx}: unaudited model calls")
        for a, b in zip(records, requests):
            if a["input_ids"] != b["input_ids"] or a["output_ids"] != b["output_ids"]:
                raise ValueError(f"Task {idx}: request/trajectory mismatch")
        if stats["generated_tokens"] != sum(len(r["output_ids"]) for r in requests):
            raise ValueError(f"Task {idx}: missing generated-token cost")
        budget = manifests[0]["config"]["actor_rollout_ref"]["rollout"]["plugin"]["val_response_length"]
        if stats["main_len"] > budget or stats["main_len"] != sum(stats[k] for k in
                ("generated_tokens", "observation_tokens", "instruction_tokens")):
            raise ValueError(f"Task {idx}: inconsistent budget")
        if trajectory["baseline_protocol"] != manifests[0]["baseline_protocol"]:
            raise ValueError(f"Task {idx}: inconsistent baseline provenance")
        for key in counts:
            counts[key] += stats.get(key, 0)
    quality = degeneration_stats(root / f"requests-{r['source_index']}.jsonl" for r in results)
    require_generation_quality(quality)
    checks = {
        "memory_was_called": counts["memory_requests"] > 0,
        "memory_was_updated": counts["memory_updates"] > 0,
        "memory_nodes_created": counts["memory_nodes"] > 0,
        "memory_format_valid": counts["invalid_memory"] == 0,
        "tool_format_recovered": summary["format_retry_failures"] == 0,
    }
    if method == "amem":
        checks["semantic_retrieval_exercised"] = counts["embedding_calls"] > 0
    report = dict(method=method, summary=summary, costs=counts, checks=checks, tasks=tasks,
                  passed=all(checks.values()), scope="integration smoke, not benchmark performance")
    (root / "memory-smoke-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    result = audit(parser.parse_args().run_root)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 2)
