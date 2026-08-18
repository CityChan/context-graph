import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from envs import discoverybench_eval
from envs.discoverybench_env import load_metadata, load_prediction
from envs.discoverybench_loader import load_discoverybench_tasks
from scripts.make_discoverybench_data import _row
from scripts.summarize_discoverybench_results import summarize


def _write_fixture(root: Path) -> None:
    task_dir = root / "discoverybench" / "real" / "test" / "demo"
    task_dir.mkdir(parents=True)
    # HF and GitHub currently disagree on punctuation for at least one real
    # test filename. Exercise the compatibility resolver here.
    (task_dir / "measurements_data.csv").write_text(
        "group,value\nA,1\nB,2\n", encoding="utf-8"
    )
    metadata = {
        "id": 999,  # filename id is authoritative in the official snapshot
        "domain": "science",
        "datasets": [{
            "name": "measurements-data.csv",
            "description": "Group measurements",
            "columns": {"raw": [
                {"name": "group", "description": "Group label"},
                {"name": "value", "description": "Measured value"},
            ]},
        }],
        "queries": [[{
            "qid": 3,
            "question_type": "relationship",
            "question": "Which group has the larger value?",
        }]],
    }
    (task_dir / "metadata_7.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    answer_dir = root / "eval"
    answer_dir.mkdir()
    with (answer_dir / "answer_key_real.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("dataset", "metadataid", "query_id", "gold_hypo")
        )
        writer.writeheader()
        writer.writerow({
            "dataset": "demo",
            "metadataid": 7,
            "query_id": 3,
            "gold_hypo": "Group B has the larger value (2 versus 1).",
        })


def test_loader_builds_gold_hidden_test_task(tmp_path):
    _write_fixture(tmp_path)
    tasks = load_discoverybench_tasks(str(tmp_path), "real", "test")

    assert len(tasks) == 1
    task = tasks[0]
    assert task["task_id"] == "real:demo:m7:q3"
    assert task["input_rel_paths"] == ["measurements-data.csv"]
    assert Path(task["input_files"][0]).name == "measurements_data.csv"
    assert task["gold_hypothesis"].startswith("Group B")
    assert task["gold_hypothesis"] not in task["instruction"]
    assert "pred_results/discovery_result.json" in task["instruction"]


def test_parquet_row_serializes_heterogeneous_metadata_as_json(tmp_path):
    _write_fixture(tmp_path)
    task = load_discoverybench_tasks(str(tmp_path), "real", "test")[0]
    task["metadata"]["empty_nested_struct"] = {"element": {}}

    row = _row(task)
    assert isinstance(row["extra_info"]["metadata"], str)
    assert load_metadata(row["extra_info"]["metadata"])["empty_nested_struct"] == {
        "element": {}
    }

    pytest.importorskip("pyarrow", reason="Parquet round-trip requires pyarrow")
    output = tmp_path / "discoverybench.parquet"
    pd.DataFrame([row]).to_parquet(output, index=False)
    stored = pd.read_parquet(output).iloc[0]["extra_info"]

    assert isinstance(stored["metadata"], str)
    assert load_metadata(stored["metadata"])["empty_nested_struct"] == {
        "element": {}
    }


def test_metadata_loader_accepts_legacy_dicts():
    metadata = {"datasets": [], "element": {}}
    assert load_metadata(metadata) is metadata


def test_prediction_contract_rejects_empty_workflow(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({"hypothesis": "B is larger", "workflow": ""}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="workflow"):
        load_prediction(path)


def test_hms_exact_match_does_not_need_api_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
    score = discoverybench_eval.score_hypothesis(
        query="Which group is larger?",
        gold_hypothesis="Group B is larger.",
        gold_workflow="",
        predicted_hypothesis="Group B is larger!",
        predicted_workflow="Compared group means.",
        metadata={"datasets": []},
        dataset_type="real",
    )
    assert score["final_score"] == 1.0
    assert score["exact_match_shortcut"] is True


def test_hms_uses_context_variable_relation_product(monkeypatch):
    replies = iter([
        {"sub_hypotheses": [{
            "text": "Gold", "context": "Adults", "variables": ["x"],
            "relation": "positive",
        }]},
        {"sub_hypotheses": [{
            "text": "Pred", "context": "adults", "variables": ["x"],
            "relation": "positive association",
        }]},
        {"size_gold": 1, "size_pred": 1, "intersection": 1},
        {"score": 0.5, "explanation": "compatible but more general"},
    ])
    monkeypatch.setattr(discoverybench_eval, "_chat_json", lambda *args, **kwargs: next(replies))
    score = discoverybench_eval.score_hypothesis(
        query="What is related?",
        gold_hypothesis="Gold text",
        gold_workflow="",
        predicted_hypothesis="Pred text",
        predicted_workflow="analysis",
        metadata={"datasets": []},
        dataset_type="real",
        client=object(),
        model="fake-judge",
    )
    assert score["recall_context"] == 1.0
    assert score["mean_accuracy_score"] == 0.5
    assert score["final_score"] == 0.5


def test_discoverybench_scripts_wire_all_three_agents_and_real_hms():
    smoke = Path(
        "scripts/smoke_discoverybench_qwen3_30b_instruct_4node.sh"
    ).read_text(encoding="utf-8")
    full = Path(
        "scripts/eval_discoverybench_qwen3_30b_instruct_8node.sh"
    ).read_text(encoding="utf-8")
    shared = Path(
        "scripts/eval_sab_react_30b_instruct_8node_smoke.sh"
    ).read_text(encoding="utf-8")

    for method in ("react", "fold", "ctxgraph"):
        assert f"  {method})" in smoke
    assert "DISCOVERYBENCH_REAL_EVAL=${DISCOVERYBENCH_REAL_EVAL:-1}" in smoke
    assert "DISCOVERYBENCH_PREWARM_PACKAGES:-numpy,pandas" in smoke
    assert "deepchem" not in smoke.split("DISCOVERYBENCH_PREWARM_PACKAGES:-", 1)[1].split("}", 1)[0]
    assert "DISCOVERYBENCH_VAL_MAX_SAMPLES:-239" in full
    assert 'python -m "$CODE_BENCHMARK_TRAIN_MODULE"' in shared


def test_result_summary_reports_mean_and_judge_errors(tmp_path):
    (tmp_path / "a.json").write_text(
        json.dumps({"score": {"final_score": 0.25}}), encoding="utf-8"
    )
    (tmp_path / "b.json").write_text(
        json.dumps({"score": {"final_score": 0.75}}), encoding="utf-8"
    )
    (tmp_path / "c.json").write_text(
        json.dumps({"score": None, "detail": "HMS judge failed: timeout"}),
        encoding="utf-8",
    )
    result = summarize(tmp_path)
    assert result["audit_records"] == 3
    assert result["hms_scored"] == 2
    assert result["judge_errors"] == 1
    assert result["mean_hms"] == 0.5
