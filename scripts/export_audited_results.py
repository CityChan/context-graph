"""Publish compact, locally reconciled evaluation records without transcripts.

Audit means file/provenance/count consistency, not independently rerun judging
or an official benchmark certification. Incomplete/error runs remain exclusions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluation_records import read_evaluation


PLUGIN_FIELDS = (
    "max_turn", "max_session", "val_max_session", "session_timeout", "branch_len",
    "turn_max_new_tokens", "process_reward", "max_traj", "must_finish", "must_search",
    "double_check", "enable_summary", "final_answer_reserve", "final_answer_safety_margin",
    "structured_graph_controller", "controller_owned_tool_formatting", "controller_action_policy",
    "contextgraph_memory_mode", "consolidation_interval", "auto_prune_max_active",
    "structured_memory_enabled", "diagnostic_fix", "search_topk_cap", "search_snippet_words",
    "search_snippet_chars", "open_page_words", "open_page_chars", "lambda_cost", "lambda_compact",
)
STATS_FIELDS = (
    "natural_finish", "forced_finish", "finalizer_attempted", "hit_timeout", "hit_token_limit",
    "hit_max_turn", "no_finish", "is_branch", "traj_num", "main_context_tokens", "total_token",
    "working_context_limit", "judge_strict_em", "judge_llm", "judge_parse_failure",
    "graph_explicit_ops", "graph_invalid_ops", "consol_controller_errors",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_revision(model_path):
    path = model_path.replace("\\", "/")
    return path.rsplit("/snapshots/", 1)[1] if "/snapshots/" in path else None


def audit_run(root):
    manifests, rows, summary = read_evaluation(root)
    stored = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    for key, value in summary.items():
        if key == "benchmark" and key not in stored:  # Older BC-P format.
            continue
        if stored.get(key) != value:
            raise ValueError(f"Saved summary disagrees with rows: {key}")
    if summary["execution_errors"] or summary["judge_parse_failures"]:
        raise ValueError("Execution or judge parse failures; not a clean score")
    first = manifests[0]
    rollout = first["config"]["actor_rollout_ref"]["rollout"]
    plugin = rollout["plugin"]
    # Publish only explicitly selected metadata, not full config or raw answers.
    result = {
        "run_id": root.name, "summary": summary,
        "model": first["model"],
        "checkpoint_revision": checkpoint_revision(first["model_path"]),
        "code_commit": first["commit"], "dataset_sha256": first["source_sha256"],
        "retrieval": first.get("retrieval", "local_bcp_corpus"),
        "seed": first["seed"], "judge_model": first["judge_model"],
        "transformers": first.get("transformers"),
        "indices": first["indices"],
        "configuration_sha256": hashlib.sha256(json.dumps(first["config"], sort_keys=True).encode()).hexdigest(),
        "protocol": {
            "prompt_length": rollout["prompt_length"], "response_length": rollout["response_length"],
            "plugin": {key: plugin[key] for key in PLUGIN_FIELDS if key in plugin},
            "chat_template": {key: plugin.get("apply_chat_template_kwargs", {}).get(key) for key in ("enable_thinking", "preserve_thinking")},
        },
        "source_files": {path.name: digest(path) for path in sorted(root.glob("manifest-*.json")) + sorted(root.glob("results-*.jsonl")) + [root / "summary.json"]},
        "rows": [{
            "source_index": row["source_index"], "task_reward": row["task_reward"],
            "is_finish": row["is_finish"], "status": row["status"],
            "stats": {key: row.get("env_stats", {})[key] for key in STATS_FIELDS if key in row.get("env_stats", {})},
        } for row in rows],
    }
    return result


def audit_arm_pilot(root):
    """Reconcile the saved empty-patch ARM pilot; no official-x86 claim."""
    def read(name):
        return json.loads((root / name).read_text(encoding="utf-8"))

    manifest = read("manifest.json")
    grading = read("grading-arm/summary.json")
    generation = read("generation_summary.json")
    predictions = [json.loads(line) for line in (root / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in (root / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(predictions) != 1 or len(rows) != 1:
        raise ValueError("Expected one ARM pilot task")
    instance = predictions[0]["instance_id"]
    sha = digest(root / "predictions.jsonl")
    patch_path = root / "instances" / instance / "model.patch"
    checks = (
        grading["runtime"] == "apptainer-arm-pilot", grading["official_x86_result"] is False,
        grading["count"] == 1, grading["grading_complete"] is True,
        grading["instance_id"] == rows[0]["instance_id"] == instance,
        manifest["instance_ids"] == [instance],
        grading["dataset"] == manifest["dataset"],
        grading["harness_commit"] == manifest["harness_commit"],
        grading["predictions_sha256"] == generation["predictions_sha256"] == sha,
        grading["sif_sha256"] == rows[0]["image"]["sif_sha256"],
        generation["method"] == manifest["method"],
        generation["selected"] == generation["generated"] == 1,
        generation["generation_errors"] == 0, rows[0]["status"] == "generated",
        # Nonempty patches need detailed test reports, absent from this sync.
        grading["empty_patch"] is True, grading["resolved"] == 0,
        generation["empty_patches"] == 1, rows[0]["patch_bytes"] == 0,
        predictions[0]["model_patch"] == "", patch_path.read_bytes() == b"",
    )
    if not all(checks):
        raise ValueError("ARM evidence inconsistent or nonempty patch requires a test-report audit")
    rollout = manifest["config"]["actor_rollout_ref"]["rollout"]
    files = ["manifest.json", "grading-arm/summary.json", "generation_summary.json", "predictions.jsonl", "results.jsonl", f"instances/{instance}/model.patch"]
    return {
        "run_id": root.name, "method": manifest["method"], "model": manifest["model"],
        "code_commit": manifest["commit"],
        "checkpoint_revision": checkpoint_revision(manifest["model_path"]),
        "selected_instance_ids": [instance], "grading": grading,
        "prompt_length": rollout["prompt_length"], "response_length": rollout["response_length"],
        "seed": manifest["seed"], "instance_seed": rows[0]["seed"],
        "generation": generation,
        "stats": {key: rows[0]["env_stats"][key] for key in STATS_FIELDS if key in rows[0]["env_stats"]},
        "source_files": {name: digest(root / name) for name in files},
        "limitations": "One empty-patch ARM compatibility result on a Verified instance, not a 500-task result or official x86 evaluation. Image and dataset hashes are recorded metadata, not locally recomputed. Calibration artifacts are not present in this sync.",
    }


def export(source, destination, arm_sources=()):
    runs, excluded = [], []
    for root in sorted(source.iterdir()):
        if not root.is_dir() or not (root / "manifest-0.json").exists():
            continue
        try:
            runs.append(audit_run(root))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            # Avoid publishing local absolute paths from exception messages.
            if isinstance(exc, OSError):
                reason = "Required evidence file missing/unreadable"
            elif isinstance(exc, KeyError):
                reason = f"Required evidence field missing: {exc.args[0]}"
            else:
                reason = str(exc)
            excluded.append({"run_id": root.name, "reason": reason})
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "contextgraph.audited-evaluations.v1",
        "scope": "Local saved BC-P/local-GAIA evidence; no rerun and no official benchmark certification",
        "runs": runs, "excluded": excluded,
        "arm_pilots": [audit_arm_pilot(root) for root in arm_sources],
    }
    (destination / "evaluations.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    lines = [
        "# Audited saved evaluations", "",
        "These scores were recalculated from saved per-instance records and checked against all three shard manifests and summaries. This is an artifact-consistency audit, not a fresh evaluation or independent verification of the judge's answers.", "",
        "Full provenance, source SHA-256 hashes, selected indices, protocol settings and answer-free per-instance outcomes are in [evaluations.json](evaluations.json). Raw prompts, answers, credentials and local paths are not published.", "",
        "GAIA uses the local BrowseComp-Plus corpus, not official GAIA retrieval. Small BC-P subsets are diagnostics. Runs span different commits and settings; compare protocols before comparing scores. Historical finalizer behavior is preserved in these results; the October 1 fixes do not retroactively improve them. Missing checkpoint revisions remain unknown. These records do not contain SWE-bench Verified scores.", "",
        "| Run | Benchmark | Model | Method | Correct / evaluated | Accuracy | Context | Commit |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for run in runs:
        s, p = run["summary"], run["protocol"]
        lines.append(f'| `{run["run_id"]}` | {s["benchmark"]} | {run["model"]} | {s["method"]} | {s["task_successes"]:g}/{s["count"]} | {s["task_accuracy"]:.2%} | {p["prompt_length"] + p["response_length"]} | `{run["code_commit"][:7]}` |')
    lines += ["", "## Excluded runs", "", "Exclusions are retained to make the selection visible; they are not scored as clean evaluations.", ""]
    lines += [f'- `{row["run_id"]}`: {row["reason"]}.' for row in excluded]
    if payload["arm_pilots"]:
        lines += ["", "## SWE-bench Verified: ARM compatibility pilot", "",
                  "This uses the **Verified dataset**, but is **not an official x86 SWE-bench score**. The dataset catalog has 500 tasks; only one task was selected. Generation's saved `grading_status=pending` predates the separate completed ARM grading summary.", "",
                  "| Run | Method | Instance | Resolved | Patch | Context |", "| --- | --- | --- | ---: | --- | ---: |"]
        for pilot in payload["arm_pilots"]:
            lines.append(f'| `{pilot["run_id"]}` | {pilot["method"]} | `{pilot["selected_instance_ids"][0]}` | 0/1 | Empty | {pilot["prompt_length"] + pilot["response_length"]} |')
        lines += ["", "Prediction SHA-256, empty patch bytes, instance IDs and manifest/grading metadata agree in the synced files. No local calibration report or image/dataset payload is available for revalidation. Later 64K Lite console transcripts are not included as artifact-audited results."]
    command = "python scripts/export_audited_results.py --source outputs --destination results/audited"
    command += "".join(f" --arm-source {root.as_posix()}" for root in arm_sources)
    lines += ["", "## Reproduce from the original local artifacts", "", "```bash", command, "```", "", "The original files are required to reproduce their hashes. The published answer-free rows suffice to recalculate the reported accuracy; they do not suffice to rerun grading.", ""]
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs"))
    parser.add_argument("--destination", type=Path, default=Path("results/audited"))
    parser.add_argument("--arm-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    payload = export(args.source, args.destination, args.arm_source)
    print(f'Audited {len(payload["runs"])} complete runs; excluded {len(payload["excluded"])} runs')
