import json

import pytest

from scripts.audit_contextgraph_ablation import audit, request_diagnostic


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
    detail = json.loads((tmp_path / "equivalence_diagnostics.json").read_text(encoding="utf-8"))
    assert detail["request_index_base"] == 0
    assert detail["tasks"][0]["main"]["tokens"]["first_difference"] == 1


def test_audit_rejects_missing_task(tmp_path):
    write_run(tmp_path)
    (tmp_path / "legacy/0.jsonl").write_text("")
    with pytest.raises(ValueError, match="coverage mismatch"):
        audit(tmp_path)


def split_run(root):
    controls, graph = root / "controls", root / "graph"
    controls.mkdir()
    graph.mkdir()
    write_run(controls)
    manifest = {"indices": [7], "commit": "test", "source_sha256": "source",
                "seed": 42, "samples": 1, "selection": "uniform"}
    (controls / "data/manifest.json").write_text(json.dumps(manifest))
    (graph / "data").mkdir()
    (graph / "data/manifest.json").write_text(json.dumps(manifest))
    for name in ("legacy", "repaired"):
        (controls / name).rename(graph / name)
    return controls, graph


def test_split_audit_reads_both_groups_without_moving_evidence(tmp_path):
    controls, graph = split_run(tmp_path)
    result = audit(controls, graph)
    assert len(result["scores"]) == 4
    assert not result["equivalence_request_differences"]
    assert (graph / "repaired/0.jsonl").exists()
    assert (controls / "audit.json").exists()
    assert not (graph / "audit.json").exists()


@pytest.mark.parametrize("key", ["commit", "source_sha256", "indices", "seed", "samples", "selection"])
def test_split_audit_rejects_incompatible_manifests(tmp_path, key):
    controls, graph = split_run(tmp_path)
    path = graph / "data/manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=key):
        audit(controls, graph)


def test_split_audit_rejects_duplicate_variant(tmp_path):
    controls, graph = split_run(tmp_path)
    (graph / "foldagent").mkdir()
    with pytest.raises(ValueError, match="exactly one directory"):
        audit(controls, graph)


def snapshot(messages, tokens=None, max_len=100):
    return {"messages": messages, "input_ids": tokens or [1], "max_len": max_len,
            "completion_kwargs": {}}


@pytest.mark.parametrize("role,category", [
    ("assistant", "appended_assistant_message_differs"),
    ("tool", "appended_observation_or_message_differs"),
])
def test_second_request_divergence_identifies_new_message(role, category):
    initial = [{"role": "user", "content": "question"}]
    a = initial + [{"role": role, "content": "x" * 1000 + "LEFT"}]
    b = initial + [{"role": role, "content": "x" * 1000 + "RIGHT"}]
    result = request_diagnostic([snapshot(initial), snapshot(a)], [snapshot(initial), snapshot(b)])
    assert result["first_different_request"] == 1
    assert result["previous_request_equal"]
    assert result["category"] == category
    assert result["messages"]["first_content_character_difference"] == 1000
    assert "LEFT" in result["messages"]["content_windows"][0]
    assert "RIGHT" in result["messages"]["content_windows"][1]


def test_same_chat_different_tokens_and_budget():
    chat = [{"role": "user", "content": "question"}]
    result = request_diagnostic([snapshot(chat, [1, 2])], [snapshot(chat, [1, 3], 200)])
    assert result["category"] == "same_messages_request_fields_differ"
    assert result["tokens"]["first_difference"] == 1
    assert result["different_fields"] == ["input_ids", "max_len"]


def test_rewritten_history_is_not_called_new_model_output():
    initial = [{"role": "assistant", "content": "old"}]
    changed = [{"role": "assistant", "content": "rewritten"}]
    result = request_diagnostic([snapshot(initial), snapshot(initial)],
                                [snapshot(initial), snapshot(changed)])
    assert result["category"] == "history_rewrite_or_message_alignment_differs"


def test_request_count_difference():
    result = request_diagnostic([snapshot([])], [snapshot([]), snapshot([])])
    assert result["first_different_request"] == 1
    assert result["category"] == "request_count_difference"


def test_matching_run_has_empty_diagnostics(tmp_path):
    write_run(tmp_path)
    assert audit(tmp_path)["diagnostic_categories"] == {}


def test_branch_only_divergence_is_diagnosed(tmp_path):
    write_run(tmp_path)
    path = tmp_path / "equivalent/0.jsonl"
    row = json.loads(path.read_text())
    row["branch_model_contexts"] = {"#0-extra": [snapshot([])]}
    path.write_text(json.dumps(row) + "\n")
    result = audit(tmp_path)
    assert result["diagnostic_categories"] == {"matching_requests": 1}
    detail = json.loads((tmp_path / "equivalence_diagnostics.json").read_text(encoding="utf-8"))
    assert detail["tasks"][0]["branch_keys_only_equivalent"] == ["#0-extra"]
