import asyncio
import copy
import json

import httpx
import pytest

from agents.gram_prompts import ENTITIES
from scripts.probe_gram_helper import FOOTER, probe, read_cases, relation_variants, variants


def saved_event():
    return {"kind": "memory_call", "operation": "entities", "response": "[]",
            "messages": [{"role": "system", "content": "Original extractor"},
                         {"role": "user", "content": json.dumps({
                             "question": "Unknown prize?", "requested_facts": "Alice was born in Paris.",
                             "document": {"id": "source", "title": "Source", "text": "Alice was born in Paris." + FOOTER},
                             "existing_entities": []})}]}


def test_ablations_change_one_input_factor_without_touching_saved_event():
    event = saved_event()
    original = copy.deepcopy(event)
    rows = list(variants(event))
    assert len(rows) == 9 and rows[0][1] == event["messages"]
    baseline = json.loads(event["messages"][-1]["content"])
    for name, messages, structured in rows[1:4]:
        payload = json.loads(messages[-1]["content"])
        expected = copy.deepcopy(baseline)
        if name == "without_tool_footer":
            expected["document"]["text"] = "Alice was born in Paris."
        else:
            expected.pop(name.removeprefix("without_"))
        assert payload == expected and messages[0] == event["messages"][0] and not structured
    assert rows[4][1][0]["content"] == ENTITIES
    assert rows[4][1][-1] == event["messages"][-1]
    assert rows[4][1] == rows[5][1] and not rows[4][2] and rows[5][2]
    assert rows[6][1] == event["messages"] and rows[6][2]
    expected = copy.deepcopy(baseline)
    expected.pop("question")
    assert json.loads(rows[7][1][-1]["content"]) == expected
    assert rows[7][1][0]["content"] == ENTITIES
    assert rows[7][1] == rows[8][1] and not rows[7][2] and rows[8][2]
    assert event == original


def test_relation_ablations_hold_document_and_entities_fixed():
    event = saved_event()
    original = copy.deepcopy(event)
    rows = list(relation_variants(event, ["Alice", "Paris"]))
    with_question = json.loads(rows[0][1][-1]["content"])
    without_question = json.loads(rows[1][1][-1]["content"])
    assert with_question.pop("question") == "Unknown prize?"
    assert with_question == without_question
    assert without_question["entities"] == ["Alice", "Paris"]
    assert rows[0][1][0] == rows[1][1][0]
    assert event == original


@pytest.mark.parametrize("behavior", ["working", "empty", "hallucination", "wrong_relation"])
@pytest.mark.parametrize("saved_question", [True, False])
def test_probe_replays_saved_requests_and_rejects_empty_or_hallucinating_helper(tmp_path, monkeypatch, behavior, saved_question):
    source = tmp_path / "old/evaluation/instances/hash/attempt-x"
    source.mkdir(parents=True)
    event = saved_event()
    if not saved_question:
        payload = json.loads(event["messages"][-1]["content"])
        payload.pop("question")
        event["messages"][-1]["content"] = json.dumps(payload)
    (source / "trajectory.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    old_bytes = (source / "trajectory.jsonl").read_bytes()
    assert len(read_cases(tmp_path / "old")) == 1
    requests = []
    def reply(request):
        body = json.loads(request.content)
        requests.append(body)
        payload = json.loads(body["messages"][-1]["content"])
        if body["messages"][0]["content"] == "Original extractor" or behavior == "empty":
            value = []
        elif payload["document"]["text"] == "..." and behavior != "hallucination":
            value = []
        elif "entities" in payload:
            value = [["Alice", "won_prize" if behavior == "wrong_relation" else "born_in", "Paris"]]
        else:
            value = ["Alice", "Paris"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})
    real_client = httpx.AsyncClient
    clients = []
    def client(**kwargs):
        value = real_client(transport=httpx.MockTransport(reply), **kwargs)
        clients.append(value)
        return value
    monkeypatch.setattr(httpx, "AsyncClient", client)
    output = tmp_path / "probe"
    status = asyncio.run(probe(tmp_path / "old", output, "http://helper", "frozen"))
    assert status == (0 if behavior == "working" else 2)
    report = json.loads((output / "helper-probe.json").read_text())
    assert report["checks_passed"] == (behavior == "working")
    assert len(report["cases"][0]["trials"]) == 9
    assert report["cases"][0]["production_calls"]
    assert report["control_calls"]
    if behavior == "working":
        assert len(report["cases"][0]["relation_trials"]) == 2
        assert all(trial["count"] == 1 for trial in report["cases"][0]["relation_trials"])
    assert requests[0]["messages"] == event["messages"]
    assert "structured_outputs" not in requests[0]
    assert all(c.is_closed for c in clients)
    assert (source / "trajectory.jsonl").read_bytes() == old_bytes
    assert len((output / "helper-probe.jsonl").read_text().splitlines()) >= 10
    with pytest.raises(FileExistsError):
        asyncio.run(probe(tmp_path / "old", output, "http://helper", "frozen"))


@pytest.mark.parametrize("failure", ["empty_entities", "unknown_endpoint"])
def test_rl_preflight_rejects_valid_but_bad_helper_output(tmp_path, monkeypatch, failure):
    from scripts.gram_rl_smoke import check_memory_schema
    real_client = httpx.AsyncClient
    count = 0
    def reply(request):
        nonlocal count
        count += 1
        value = [] if failure == "empty_entities" else ["Alice", "Paris"] if count == 1 else [["Alice", "born_in", "Paris"], ["Alice", "knows", "Unknown"]]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(reply), **kwargs))
    with pytest.raises(ValueError, match="semantic preflight"):
        asyncio.run(check_memory_schema(tmp_path, "http://helper", "frozen"))
    assert not (tmp_path / "memory-schema-preflight.json").exists()
    assert (tmp_path / "memory-schema-preflight.jsonl").is_file()
