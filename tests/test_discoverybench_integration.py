import csv
import json
from types import SimpleNamespace
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
    full_8b = Path(
        "scripts/eval_discoverybench_qwen3_8b_4node.sh"
    ).read_text(encoding="utf-8")
    submit_8b = Path(
        "scripts/submit_eval_discoverybench_qwen3_8b_4node.sh"
    ).read_text(encoding="utf-8")
    submit_30b = Path(
        "scripts/submit_eval_discoverybench_qwen3_30b_instruct_8node.sh"
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
    assert "DISCOVERYBENCH_VAL_MAX_TURN:-24" in full
    assert "#SBATCH -t 02:00:00" in full
    assert "#SBATCH -N 4" in full_8b
    assert "MODEL_PATH:-Qwen/Qwen3-8B" in full_8b
    assert "DISCOVERYBENCH_VAL_MAX_SAMPLES:-239" in full_8b
    assert "SAB_ROLLOUT_QUANTIZATION:-none" in full_8b
    assert 'if [ "$SAB_ROLLOUT_QUANTIZATION" != "none" ]' in shared
    for method in ("react", "fold", "ctxgraph"):
        assert method in submit_8b
        assert method in submit_30b
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in submit_30b
    assert 'nodes=8' in submit_30b
    assert "DISCOVERYBENCH_VAL_MAX_SAMPLES:-239" in submit_30b
    assert "DISCOVERYBENCH_VAL_MAX_TURN:-24" in submit_30b
    assert "DISCOVERYBENCH_TIME_LIMIT:-02:00:00" in submit_30b
    assert "DISCOVERYBENCH_VAL_MAX_TURN=$DISCOVERYBENCH_VAL_MAX_TURN" in submit_30b
    assert "sed -nE 's/^([0-9]+)(;[^[:space:]]+)?$/\\1/p'" in submit_30b
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


def _completion_response(payload, *, finish_reason="stop", usage=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=json.dumps(payload), refusal=None),
        )],
        usage=usage,
    )


def test_judge_uses_current_completion_token_parameter(monkeypatch):
    monkeypatch.delenv("DISCOVERYBENCH_JUDGE_MAX_COMPLETION_TOKENS", raising=False)
    monkeypatch.delenv("DISCOVERYBENCH_JUDGE_REASONING_EFFORT", raising=False)
    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _completion_response({"ok": True})

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert discoverybench_eval._chat_json(client, "gpt-5-nano", "prompt") == {"ok": True}
    assert completions.calls[0]["max_completion_tokens"] == 4096
    assert completions.calls[0]["reasoning_effort"] == "minimal"
    assert "max_tokens" not in completions.calls[0]


def test_judge_falls_back_for_legacy_completion_token_parameter(monkeypatch):
    monkeypatch.delenv("DISCOVERYBENCH_JUDGE_MAX_COMPLETION_TOKENS", raising=False)
    monkeypatch.delenv("DISCOVERYBENCH_JUDGE_REASONING_EFFORT", raising=False)
    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "max_completion_tokens" in kwargs:
                raise RuntimeError("Unsupported parameter: max_completion_tokens")
            return _completion_response({"ok": True})

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert discoverybench_eval._chat_json(client, "legacy", "prompt") == {"ok": True}
    assert "max_completion_tokens" in completions.calls[0]
    assert completions.calls[1]["max_tokens"] == 4096
    assert "reasoning_effort" not in completions.calls[0]


def test_judge_drops_unsupported_reasoning_effort(monkeypatch):
    monkeypatch.delenv("DISCOVERYBENCH_JUDGE_REASONING_EFFORT", raising=False)

    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "reasoning_effort" in kwargs:
                raise RuntimeError("Unsupported parameter: reasoning_effort")
            return _completion_response({"ok": True})

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert discoverybench_eval._chat_json(client, "gpt-5-nano", "prompt") == {"ok": True}
    assert completions.calls[0]["reasoning_effort"] == "minimal"
    assert "reasoning_effort" not in completions.calls[1]


def test_judge_reports_empty_completion_details(monkeypatch):
    monkeypatch.delenv("DISCOVERYBENCH_JUDGE_REASONING_EFFORT", raising=False)

    class Completions:
        def create(self, **kwargs):
            del kwargs
            usage = SimpleNamespace(
                completion_tokens=4096,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=4096),
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content="", refusal=None),
                )],
                usage=usage,
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    with pytest.raises(discoverybench_eval.DiscoveryBenchJudgeError) as exc_info:
        discoverybench_eval._chat_json(client, "gpt-5-nano", "prompt", retries=1)
    message = str(exc_info.value)
    assert "judge returned empty content" in message
    assert "finish_reason='length'" in message
    assert "reasoning_tokens=4096" in message


def test_judge_drops_unsupported_temperature():
    class Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(dict(kwargs))
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported parameter: temperature")
            return _completion_response({"ok": True})

    completions = Completions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert discoverybench_eval._chat_json(client, "gpt-5-nano", "prompt") == {"ok": True}
    assert "temperature" in completions.calls[0]
    assert "temperature" not in completions.calls[1]
