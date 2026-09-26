import json
import subprocess
import sys
from pathlib import Path

import pytest

from agents.diagnostic_fixes import RepeatAdvice, validate_fix


def test_repeat_search_is_advisory_and_once_per_exact_query():
    advice = RepeatAdvice()
    assert not advice.observe("search", {"query": "one"})
    assert not advice.observe("search", {"query": "two"})
    assert not advice.observe("search", {"query": "one"})
    assert "[REPEATED ACTION]" in advice.observe("search", {"query": "one"})
    assert not advice.observe("search", {"query": "one"})
    assert advice.warnings == 1
    assert not RepeatAdvice().observe("search", {"query": "one"})


def test_noop_focus_is_counted_separately():
    advice = RepeatAdvice()
    for _ in range(2):
        assert not advice.observe("select_noop", {"node_id": "n1"})
    assert advice.observe("select_noop", {"node_id": "n1"})
    assert not advice.observe("select_noop", {"node_id": "n2"})
    for _ in range(3):
        assert not advice.observe("open", {"docid": "1"})


def test_fix_validation():
    assert validate_fix("none", True) == "none"
    assert validate_fix("answer", False) == "answer"
    with pytest.raises(ValueError):
        validate_fix("answer", True)
    with pytest.raises(ValueError):
        validate_fix("typo", False)


def test_pair_summary_checks_provenance_and_uses_task_reward(tmp_path):
    for variant, fix, score in (("repaired", "none", 0), ("answer", "answer", 1)):
        directory = tmp_path / variant
        directory.mkdir()
        row = dict(agent_name="main", task_id="test", model_contexts=[{}],
                   branch_model_contexts={}, diagnostic_fix=fix, task_reward=score,
                   reward=1.1)
        (directory / "0.jsonl").write_text(json.dumps(row), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/summarize_contextgraph_fix.py"
    subprocess.run([sys.executable, str(script), str(tmp_path), "answer"], check=True, capture_output=True)
    report = json.loads((tmp_path / "fix_audit.json").read_text())
    assert report["scores"]["repaired"]["successes"] == 0
    assert report["scores"]["answer"]["successes"] == 1
    assert report["changed_tasks"] == [{"task_id": "test", "baseline": 0, "intervention": 1}]
    row["diagnostic_fix"] = "none"
    (tmp_path / "answer/0.jsonl").write_text(json.dumps(row), encoding="utf-8")
    assert subprocess.run([sys.executable, str(script), str(tmp_path), "answer"], capture_output=True).returncode != 0
