import asyncio
import json

import httpx
import pytest
from jsonschema import Draft202012Validator

from agents.gram_agent import MemoryBackend, MemoryBackendError
from agents.gram_memory import GraphMemory, memory_output_schema


def test_relation_schema_binds_only_endpoints_to_actual_extracted_entities():
    names = ['Alice "A"', "Paris", "北京"]
    schema = memory_output_schema("relations", entities=names)
    Draft202012Validator.check_schema(schema)
    check = Draft202012Validator(schema)
    assert check.is_valid([])
    assert check.is_valid([['Alice "A"', "born_in", "Paris"], ["Paris", "linked_to", "北京"]])
    for invalid in ([['Alice', "born_in", "Paris"]], [['Alice "A"', "born_in", "London"]],
                    [['Alice "A"', "Paris"]], [['Alice "A"', "born_in", "Paris", "extra"]]):
        assert not check.is_valid(invalid)
    empty = Draft202012Validator(memory_output_schema("relations", entities=[]))
    assert empty.is_valid([]) and not empty.is_valid([["A", "r", "B"]])


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
            assert body["structured_outputs"]["json"] == memory_output_schema(operation,
                entities=["Alice", "Paris"] if operation == "relations" else None)
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
    requests, clients = [], []
    def reply(request):
        body = json.loads(request.content)
        requests.append(body)
        payload = json.loads(body["messages"][-1]["content"])
        assert "question" not in payload
        operation = "maintenance" if "graph" in payload else "relations" if "entities" in payload else "entities"
        assert body["structured_outputs"]["json"] == memory_output_schema(operation,
            entities=["Alice", "Paris"] if operation == "relations" else None)
        if operation == "maintenance":
            value = {"add": [], "remove": []}
        elif payload["document"]["text"] == "...":
            value = []
        elif operation == "entities":
            value = ["Alice", "Paris"]
        else:
            value = [["Alice", "born_in"]] if bad else [["Alice", "born_in", "Paris"]]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})
    real_client = httpx.AsyncClient
    def client(**kwargs):
        result = real_client(**kwargs, transport=httpx.MockTransport(reply))
        clients.append(result)
        return result
    monkeypatch.setattr(httpx, "AsyncClient", client)
    if bad:
        with pytest.raises(ValueError, match="semantic preflight.*three strings"):
            asyncio.run(check_memory_schema(tmp_path, "http://helper", "frozen"))
        assert not (tmp_path / "memory-schema-preflight.json").exists()
    else:
        asyncio.run(check_memory_schema(tmp_path, "http://helper", "frozen"))
        assert json.loads((tmp_path / "memory-schema-preflight.json").read_text())["checked"] == operations
    assert clients[0].is_closed
    rows = [json.loads(row) for row in (tmp_path / "memory-schema-preflight.jsonl").read_text().splitlines()]
    calls = [row for row in rows if row["kind"] == "memory_call"]
    assert len(calls) == (3 if bad else 4)
    assert all(row["response"] and row["structured_outputs"] for row in calls)
    assert (tmp_path / "memory-semantic-preflight.json").is_file()


@pytest.mark.parametrize("operation", ["memory_insert", "memory_update"])
def test_research_question_cannot_change_any_production_helper_request(operation):
    async def run():
        requests = []
        backend = MemoryBackend("http://memory", "frozen")
        await backend.client.aclose()
        def reply(request):
            body = json.loads(request.content)
            requests.append(body)
            payload = json.loads(body["messages"][-1]["content"])
            assert "question" not in payload
            assert payload["requested_facts"] == "birthplace"
            assert payload["document"]["text"] == "Alice was born in Paris."
            if operation == "memory_update":
                value = {"add": [["Alice", "born_in", "Paris"]], "remove": []}
            else:
                value = [["Alice", "born_in", "Paris"]] if "entities" in payload else ["Alice", "Paris"]
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})
        backend.client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        try:
            for question in ("Where was Alice born?", "Which UNKNOWN_PRIZE did Alice win?"):
                graph = GraphMemory()
                await backend.edit(operation, "birthplace", {"id": "source", "title": "Source", "text": "Alice was born in Paris."},
                                   question, graph, .9)
                assert ("Alice", "born_in", "Paris") in graph.edges
            half = len(requests) // 2
            assert requests[:half] == requests[half:]
        finally:
            await backend.aclose()
    asyncio.run(run())
