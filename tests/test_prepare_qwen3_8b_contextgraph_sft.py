import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "prepare_qwen3_8b_contextgraph_sft.py"
SPEC = importlib.util.spec_from_file_location("contextgraph_test_prepare_qwen3_sft", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(trajectory_id: str, task_id: str, domain: str) -> dict:
    return {
        "messages": [{"role": "assistant", "content": trajectory_id}],
        "tools": [],
        "enable_thinking": False,
        "trajectory_id": trajectory_id,
        "task_id": task_id,
        "domain": domain,
        "task_reward": 1.0,
        "graph_structural_ops": 1,
        "graph_invalid_ops": 0,
        "graph_trace_json": "{}",
    }


def test_assembly_deduplicates_and_keeps_tasks_in_one_split(tmp_path, monkeypatch):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    frames = {
        first: pd.DataFrame([row("a", "1", "alfworld"), row("b", "2", "alfworld")]),
        second: pd.DataFrame([row("a", "1", "alfworld"), row("c", "3", "alfworld")]),
    }
    monkeypatch.setattr(MODULE.pd, "read_parquet", lambda path: frames[path].copy())
    train, validation, summary = MODULE.assemble_dataset([first, second], validation_fraction=0.34, seed=42)
    assert len(train) + len(validation) == 3
    assert summary["duplicate_trajectories_removed"] == 1
    train_keys = set(zip(train.domain, train.task_id))
    validation_keys = set(zip(validation.domain, validation.task_id))
    assert train_keys.isdisjoint(validation_keys)


def test_quality_gates_reject_invalid_graph_rows():
    frame = pd.DataFrame([row("bad", "1", "alfworld")])
    frame.loc[0, "graph_invalid_ops"] = 1
    try:
        MODULE.validate_rows(frame)
    except ValueError as exc:
        assert "invalid_ops" in str(exc)
    else:
        raise AssertionError("invalid graph row was accepted")
