"""Code-execution environment and HMS reward for DiscoveryBench."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from envs.discoverybench_eval import score_hypothesis
from envs.scienceagent_env import ScienceAgentEnv


def load_prediction(path: str | Path) -> tuple[str, str]:
    """Read and validate the benchmark's agent-facing result contract."""
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("discovery_result.json must contain one JSON object")
    hypothesis = value.get("hypothesis")
    workflow = value.get("workflow")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("`hypothesis` must be a non-empty string")
    if not isinstance(workflow, str) or not workflow.strip():
        raise ValueError("`workflow` must be a non-empty string")
    return hypothesis.strip(), workflow.strip()


def _safe_task_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_") or "unknown"


class DiscoveryBenchEnv(ScienceAgentEnv):
    """Reuse the persistent Python sandbox, replacing SAB artifact scoring."""

    async def init_env(self, item):
        await super().init_env(item)
        if self.env_fail:
            return
        extra = self.instance_info
        self.query = str(extra.get("query", ""))
        self.gold_hypothesis = str(extra.get("gold_hypothesis", ""))
        self.gold_workflow = str(extra.get("gold_workflow", ""))
        self.discovery_metadata = extra.get("metadata", {}) or {}
        self.dataset_type = str(extra.get("dataset_type", "real"))
        self.eval_contract = (
            "Write UTF-8 JSON, not Markdown or JSONL. The root must be exactly one "
            "object with two non-empty string fields:\n"
            '{"hypothesis": "direct quantitative answer", '
            '"workflow": "analysis steps and statistical evidence used"}\n'
            "The evaluator scores hypothesis meaning; merely producing valid JSON "
            "does not earn HMS credit."
        )

    async def get_reward(self, item, messages, context):
        if self.env_fail or self.sandbox is None or not self.workdir:
            return "DiscoveryBench environment unavailable", 0.0, {}

        produced = self.sandbox.list_output_files()
        metrics: dict[str, Any] = {
            "produced_files": len(produced),
            "valid_result_json": 0,
        }
        result_path = Path(self.workdir) / "pred_results" / "discovery_result.json"
        if not result_path.is_file():
            return "missing pred_results/discovery_result.json", 0.0, metrics

        try:
            hypothesis, workflow = load_prediction(result_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return f"invalid discovery_result.json: {exc}", 0.0, metrics
        metrics["valid_result_json"] = 1

        use_real = os.environ.get("DISCOVERYBENCH_REAL_EVAL", "").lower() in {
            "1", "true", "yes"
        }
        if not use_real:
            detail = "format-only smoke: valid discovery_result.json (not HMS)"
            self._write_audit(hypothesis, workflow, None, detail)
            return detail, 1.0, metrics

        try:
            score_record = await asyncio.to_thread(
                score_hypothesis,
                query=self.query,
                gold_hypothesis=self.gold_hypothesis,
                gold_workflow=self.gold_workflow,
                predicted_hypothesis=hypothesis,
                predicted_workflow=workflow,
                metadata=self.discovery_metadata,
                dataset_type=self.dataset_type,
            )
        except Exception as exc:  # noqa: BLE001 - evaluator errors become audit records
            metrics["judge_error"] = 1
            detail = f"HMS judge failed: {type(exc).__name__}: {exc}"
            print(f"[DiscoveryBench eval] task={self.task_id} {detail}")
            self._write_audit(hypothesis, workflow, None, detail)
            return detail, 0.0, metrics

        final_score = float(score_record["final_score"])
        metrics.update({
            "hms_score": final_score,
            "hms_context_recall": float(score_record["recall_context"]),
            "hms_mean_accuracy": float(score_record["mean_accuracy_score"]),
            "judge_error": 0,
        })
        detail = (
            f"HMS={final_score:.4f} context_recall="
            f"{metrics['hms_context_recall']:.4f} mean_accuracy="
            f"{metrics['hms_mean_accuracy']:.4f}"
        )
        print(f"[DiscoveryBench eval] task={self.task_id} {detail}")
        self._write_audit(hypothesis, workflow, score_record, detail)
        return detail, final_score, metrics

    def _write_audit(self, hypothesis: str, workflow: str,
                     score_record: dict | None, detail: str) -> None:
        record = {
            "task_id": self.task_id,
            "query": self.query,
            "dataset_type": self.dataset_type,
            "gold_hypothesis": self.gold_hypothesis,
            "gold_workflow": self.gold_workflow,
            "predicted_hypothesis": hypothesis,
            "predicted_workflow": workflow,
            "detail": detail,
            "score": score_record,
        }
        targets = [Path(self.workdir) / "discoverybench_eval.json"]
        result_dir = os.environ.get("DISCOVERYBENCH_RESULTS_DIR")
        if result_dir:
            directory = Path(result_dir)
            directory.mkdir(parents=True, exist_ok=True)
            targets.append(directory / f"{_safe_task_name(str(self.task_id))}.json")
        for target in targets:
            try:
                target.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                print(f"[DiscoveryBench eval] audit write failed at {target}: {exc}")
