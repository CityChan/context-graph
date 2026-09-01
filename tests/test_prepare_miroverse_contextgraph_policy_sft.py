import json
import sys

from scripts.prepare_miroverse_contextgraph_policy_sft import (
    boxed_answer,
    build_rows,
    convert_call,
    record_to_row,
    split_rows,
)


def trajectory(tool_server="browsing-agent", tool_name="search_and_browse", arguments=None):
    arguments = arguments or {"subtask": "Find the actor and cite evidence"}
    return {
        "messages": [
            {"role": "system", "content": "MiroFlow tools"},
            {"role": "user", "content": "Who played the role?"},
            {
                "role": "assistant",
                "content": (
                    "I will delegate this lookup. <use_mcp_tool>"
                    f"<server_name>{tool_server}</server_name>"
                    f"<tool_name>{tool_name}</tool_name>"
                    f"<arguments>{json.dumps(arguments)}</arguments>"
                    "</use_mcp_tool>"
                ),
            },
            {"role": "user", "content": "The actor was Example Person."},
            {"role": "assistant", "content": "Therefore, \\boxed{Example Person}."},
        ]
    }


def test_maps_browsing_agent_to_contextgraph_branch():
    row = record_to_row(trajectory(), source_subset="test")
    assert row is not None
    assert row["mapped_branch_calls"] == 1
    assert "<function=branch>" in row["messages"][2]["content"]
    assert "<parameter=prompt>Find the actor and cite evidence</parameter>" in row["messages"][2]["content"]
    assert "<function=finish>" in row["messages"][-1]["content"]
    assert "<parameter=answer>Example Person</parameter>" in row["messages"][-1]["content"]
    assert "MiroFlow tools" not in row["messages"][0]["content"]


def test_maps_google_search_and_rejects_code_tools():
    call = convert_call(
        "tool-google-search",
        "google_search",
        json.dumps({"q": "example query", "num": 5}),
    )
    assert call is not None and "<function=search>" in call and "example query" in call
    reasons = []
    row = record_to_row(
        trajectory("tool-code", "run_python_code", {"code_block": "print(1)"}),
        source_subset="test",
        rejection_reasons=reasons,
    )
    assert row is None
    assert reasons == ["unsupported_tool:tool-code/run_python_code"]


def test_rejects_non_action_intermediate_assistant():
    record = trajectory()
    record["messages"][2]["content"] = "I think the answer may be X."
    reasons = []
    assert record_to_row(record, source_subset="test", rejection_reasons=reasons) is None
    assert reasons == ["assistant_call_count"]


def test_rejects_malformed_mcp_block():
    record = trajectory()
    record["messages"][2]["content"] = "<use_mcp_tool><server_name>browsing-agent</server_name>"
    reasons = []
    assert record_to_row(record, source_subset="test", rejection_reasons=reasons) is None
    assert reasons == ["malformed_mcp_call"]


def test_boxed_answer_handles_nested_braces():
    assert boxed_answer(r"Result: \boxed{Dorothy \text{Dottie} Hinson}") == r"Dorothy \text{Dottie} Hinson"


def test_build_deduplicates_and_split_has_no_query_overlap():
    rows, counters = build_rows(
        [trajectory(), trajectory()],
        max_samples=0,
        source_subset="test",
    )
    assert len(rows) == 1
    assert counters["accepted"] == 1
    assert counters["duplicates"] == 1
    second = trajectory()
    second["messages"][1]["content"] = "A second question?"
    rows, _ = build_rows([trajectory(), second], max_samples=0, source_subset="test")
    train, validation = split_rows(rows, validation_fraction=0.5, seed=42)
    assert {row["query_hash"] for row in train}.isdisjoint(
        {row["query_hash"] for row in validation}
    )


def test_cli_writes_train_validation_and_manifest(tmp_path, monkeypatch):
    from scripts import prepare_miroverse_contextgraph_policy_sft as module

    input_path = tmp_path / "source.jsonl"
    input_path.write_text(json.dumps(trajectory()) + "\n", encoding="utf-8")
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    manifest_path = tmp_path / "manifest.json"
    writes = {}
    monkeypatch.setattr(
        module,
        "write_parquet",
        lambda rows, path: writes.__setitem__(path.name, list(rows)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--input",
            str(input_path),
            "--output",
            str(train_path),
            "--validation-output",
            str(validation_path),
            "--manifest",
            str(manifest_path),
            "--validation-fraction",
            "0",
        ],
    )
    module.main()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["accepted"] == 1
    assert manifest["train_rows"] == 1
    assert manifest["validation_rows"] == 0
    assert manifest["query_overlap"] == 0
    assert len(writes["train.parquet"]) == 1
    assert writes["validation.parquet"] == []
