import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_miroverse_contextgraph_controller_sft import (
    GraphActionController,
    convert_record,
    extract_evidence,
)


def miroverse_record():
    return {
        "split": "MiroVerse-MuSiQue",
        "query": "Which city is the birthplace of the author of Example Book?",
        "answer": "Example City",
        "messages": [
            {"role": "system", "content": "Use one tool at a time."},
            {"role": "user", "content": "Which city is the birthplace of the author of Example Book?"},
            {"role": "assistant", "content": "<search><query>Example Book author</query></search>"},
            {"role": "user", "content": "Search result: Example Book was written by Ada Example."},
            {"role": "assistant", "content": "<search><query>Ada Example birthplace</query></search>"},
            {"role": "user", "content": "Search result: Ada Example was born in Example City."},
            {"role": "assistant", "content": "The answer is Example City."},
        ],
    }


def test_extracts_user_role_tool_observations_after_assistant_calls():
    question, answer, evidence = extract_evidence(miroverse_record())
    assert question.startswith("Which city")
    assert answer == "Example City"
    assert evidence == [
        "Search result: Example Book was written by Ada Example.",
        "Search result: Ada Example was born in Example City.",
    ]


def test_converts_one_snapshot_and_replays_structural_action():
    response = json.dumps(
        {
            "action": "add_edge",
            "candidate_indices": [0, 1],
            "summary": "",
            "relation": "causal",
        }
    )
    row = convert_record(
        miroverse_record(),
        controller=GraphActionController(max_candidates=12, preview_chars=360),
        teacher_response=response,
        teacher_model="deepseek-ai/DeepSeek-V4-Flash-0731",
        source_subset="MiroVerse-MuSiQue",
        sample_index=0,
    )
    assert row["action"] == "add_edge"
    assert row["candidate_count"] == 2
    assert row["replay_valid"] is True
    assert row["before_hash"] != row["after_hash"]
    assert row["messages"][-1] == {"role": "assistant", "content": response}
    assert "Private teacher-only" not in row["messages"][-2]["content"]


def test_cli_writes_controller_parquet_with_offline_teacher(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    source = tmp_path / "miroverse.jsonl"
    source.write_text(json.dumps(miroverse_record()) + "\n", encoding="utf-8")
    output = tmp_path / "controller.parquet"
    response = json.dumps(
        {
            "action": "add_edge",
            "candidate_indices": [0, 1],
            "summary": "",
            "relation": "causal",
        }
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_miroverse_contextgraph_controller_sft.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--max-samples",
            "1",
            "--teacher-response",
            response,
        ],
    )
    from scripts.prepare_miroverse_contextgraph_controller_sft import main

    main()
    frame = pd.read_parquet(output)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert len(frame) == 1
    assert frame.iloc[0]["action"] == "add_edge"
    assert manifest["all_replay_valid"] is True
