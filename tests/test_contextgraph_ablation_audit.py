import json

import pytest

from scripts.audit_contextgraph_ablation import audit


def write_run(root, differing=False):
    (root / "data").mkdir()
    (root / "data/manifest.json").write_text(json.dumps({"indices": [7], "commit": "test"}))
    for name in ("foldagent", "equivalent", "legacy", "repaired"):
        (root / name).mkdir()
        row = {"task_id": "bcp-diagnostic-7", "agent_name": "main", "task_reward": 1,
               "model_contexts": [{"input_ids": [1, 2]}], "branch_model_contexts": {}}
        if differing and name == "equivalent":
            row["model_contexts"][0]["input_ids"] = [1, 3]
        (root / name / "0.jsonl").write_text(json.dumps(row) + "\n")


def test_audit_detects_request_divergence(tmp_path):
    write_run(tmp_path, differing=True)
    result = audit(tmp_path)
    assert result["equivalence_request_differences"][0]["first_different_request"] == 0
    assert result["scores"]["repaired"]["successes"] == 1


def test_audit_rejects_missing_task(tmp_path):
    write_run(tmp_path)
    (tmp_path / "legacy/0.jsonl").write_text("")
    with pytest.raises(ValueError, match="coverage mismatch"):
        audit(tmp_path)
