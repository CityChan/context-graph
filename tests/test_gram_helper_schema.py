import asyncio
import json

import httpx
import pytest
from jsonschema import Draft202012Validator

from agents.gram_agent import MemoryBackend, MemoryBackendError
from agents.gram_memory import GraphMemory, memory_output_schema


@pytest.mark.parametrize("stage", ["explicit_skip", "no_entities", "no_relations"])
def test_empty_helper_edits_report_stage_without_fabricating_facts(stage):
    async def run():
        backend = MemoryBackend("http://memory", "frozen")
        await backend.client.aclose()
        requests = []
        def reply(request):
            requests.append(json.loads(request.content))
            value = ["Alice", "Paris"] if stage == "no_relations" and len(requests) == 1 else []
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        graph = GraphMemory()
        try:
            result = await backend.edit("memory_insert", "None" if stage == "explicit_skip" else "unverified fact",
                                        {"id": "doc", "text": "source", "title": "Source"}, "Question?", graph, .9)
            assert result["extraction_status"] == stage
            assert not result["add"] and not graph.edges
            assert len(requests) == {"explicit_skip": 0, "no_entities": 1, "no_relations": 2}[stage]
        finally:
            await backend.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("operation,valid,invalid", [
    ("entities", [[], ["Alice", "Paris"]], [[1], [""]]),
    ("relations", [[], [["Alice", "born_in", "Paris"]]],
     [["Alice", "born_in", "Paris"], [["Alice", "born_in"]],
      [["Alice", "born_in", "Paris", "extra"]], [["Alice", 1, "Paris"]],
      [{"subject": "Alice", "relation": "born_in", "object": "Paris"}]]),
    ("maintenance", [{"add": [], "remove": []}, {"add": [["A", "r", "B"]], "remove": []}],
     [{"add": []}, {"add": [], "remove": [], "extra": []}, {"add": [["A", "r"]], "remove": []}]),
])
def test_helper_schemas_reject_malformed_triples(operation, valid, invalid):
    schema = memory_output_schema(operation)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert all(validator.is_valid(value) for value in valid)
    assert all(not validator.is_valid(value) for value in invalid)


@pytest.mark.parametrize("bad", [None, "shape", "entity", "length", "http"])
def test_helper_http_schema_audit_and_atomic_failure(bad):
    async def run():
        events, requests = [], []
        backend = MemoryBackend("http://memory", "frozen", audit=events.append)
        await backend.client.aclose()
        graph = GraphMemory()
        graph.apply([["Old", "related_to", "Fact"]], [], "old")
        original = graph.snapshot()
        def reply(request):
            body = json.loads(request.content)
            requests.append(body)
            operation = "entities" if len(requests) == 1 else "relations"
            assert body["structured_outputs"]["json"] == memory_output_schema(operation)
            assert body["chat_template_kwargs"] == {"enable_thinking": False}
            if bad == "http":
                return httpx.Response(400, json={"error": "unsupported schema"})
            value = ["Alice", "Paris"] if operation == "entities" else [["Alice", "born_in", "Paris"]]
            if operation == "relations" and bad in {"shape", "entity"}:
                value = [["Alice", "born_in"]] if bad == "shape" else [["Alice", "born_in", "Unknown"]]
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)},
                "finish_reason": "length" if bad == "length" else "stop"}]})
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        try:
            edit = backend.edit("memory_insert", "birthplace", {"id": "new", "title": "Alice", "text": "Alice born in Paris"},
                                "Where?", graph, .9)
            if bad:
                with pytest.raises((MemoryBackendError, httpx.HTTPStatusError)):
                    await edit
                assert graph.snapshot() == original
                if bad != "http":
                    assert events[-1]["response"]  # Invalid raw output is retained.
            else:
                await edit
                assert ("Alice", "born_in", "Paris") in graph.edges
                assert graph.edges[("Old", "related_to", "Fact")] == {"old"}
                assert [e["operation"] for e in events] == ["entities", "relations"]
            for event, request in zip(events, requests):
                assert event["structured_outputs"] == request["structured_outputs"]
        finally:
            await backend.aclose()
    asyncio.run(run())


def test_maintenance_schema_reaches_helper_and_preserves_provenance():
    async def run():
        backend = MemoryBackend("http://memory", "frozen")
        await backend.client.aclose()
        graph = GraphMemory()
        graph.apply([["Alice", "born_in", "London"]], [], "old")
        def reply(request):
            body = json.loads(request.content)
            assert body["structured_outputs"]["json"] == memory_output_schema("maintenance")
            value = {"remove": [["Alice", "born_in", "London"]], "add": [["Alice", "born_in", "Paris"]]}
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        try:
            await backend.edit("memory_update", "correct birthplace", {"id": "new", "text": "Alice born in Paris", "title": "Correction"},
                               "Where?", graph, .9)
            assert graph.snapshot() == [{"triple": ["Alice", "born_in", "Paris"], "sources": ["new"]}]
        finally:
            await backend.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("bad", [False, True])
def test_live_probe_checks_all_operations_or_leaves_failure_audit(tmp_path, monkeypatch, bad):
    from scripts.gram_rl_smoke import check_memory_schema
    operations = ["entities", "relations", "maintenance"]
    values = [["Alice", "Paris"], [["Alice", "born_in", "Paris"]], {"add": [], "remove": []}]
    requests, clients = [], []
    def reply(request):
        index = len(requests)
        body = json.loads(request.content)
        requests.append(body)
        assert body["structured_outputs"]["json"] == memory_output_schema(operations[index])
        value = [["Alice", "born_in"]] if bad and index == 1 else values[index]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})
    real_client = httpx.AsyncClient
    def client(**kwargs):
        result = real_client(**kwargs, transport=httpx.MockTransport(reply))
        clients.append(result)
        return result
    monkeypatch.setattr(httpx, "AsyncClient", client)
    if bad:
        with pytest.raises(MemoryBackendError, match="three strings"):
            asyncio.run(check_memory_schema(tmp_path, "http://helper", "frozen"))
        assert not (tmp_path / "memory-schema-preflight.json").exists()
    else:
        asyncio.run(check_memory_schema(tmp_path, "http://helper", "frozen"))
        assert json.loads((tmp_path / "memory-schema-preflight.json").read_text())["checked"] == operations
    assert clients[0].is_closed
    rows = [json.loads(row) for row in (tmp_path / "memory-schema-preflight.jsonl").read_text().splitlines()]
    assert len(rows) == (2 if bad else 3)
    assert all(row["response"] and row["structured_outputs"] for row in rows)
