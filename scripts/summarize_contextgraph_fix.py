"""Summarize paired evaluation-only prompt interventions using task reward."""
import argparse
import json
from pathlib import Path

from audit_contextgraph_ablation import read_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("fix", choices=["answer", "repeat"])
    args = parser.parse_args()
    rows = {v: read_main(args.root / v) for v in ("repaired", args.fix)}
    baseline, intervention = rows.values()
    if not baseline or set(baseline) != set(intervention):
        raise ValueError("Empty evaluation or task coverage mismatch")
    for variant, tasks in rows.items():
        expected = "none" if variant == "repaired" else args.fix
        if any(r.get("diagnostic_fix") != expected for r in tasks.values()):
            raise ValueError(f"Missing or incorrect diagnostic_fix provenance: {variant}")
    result = {
        "scores": {v: {"successes": sum(float(r["task_reward"]) > 0 for r in tasks.values()),
                        "samples": len(tasks),
                        "repeat_advice_count": sum(r.get("repeat_advice_count", 0) for r in tasks.values())}
                   for v, tasks in rows.items()},
        "changed_tasks": [{"task_id": k, "baseline": baseline[k]["task_reward"],
                           "intervention": intervention[k]["task_reward"]}
                          for k in sorted(baseline)
                          if baseline[k]["task_reward"] != intervention[k]["task_reward"]],
        "note": "Paired diagnostic; inspect trajectories and generation variability before causal claims.",
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    (args.root / "fix_audit.json").write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
