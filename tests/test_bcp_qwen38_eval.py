import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts.eval_bcp_qwen38 import TokenClient, completion_budget, select_indices, summarize


def test_selection_matches_across_methods_and_shards():
    indices = select_indices(150, 8, 42)
    assert indices == select_indices(150, 8, 42)
    assert sorted(sum((indices[rank::3] for rank in range(3)), [])) == indices
    assert select_indices(150, -1, 42) == list(range(150))
    with pytest.raises(ValueError):
        select_indices(150, 151, 42)


def test_budget_preserves_finalizer_override_without_exceeding_context():
    config = SimpleNamespace(prompt_length=8, response_length=24,
                             plugin=SimpleNamespace(turn_max_new_tokens=12))
    assert completion_budget(config, [1] * 10, {}) == 12
    assert completion_budget(config, [1] * 10, {"bypass_turn_max_new_tokens": True}) == 22
    assert completion_budget(config, [1] * 30, {"max_new_tokens": 1024}) == 2


def test_token_client_keeps_exact_ids_and_uses_no_judge_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "judge-secret-must-not-be-sent")
    requests = []
    def handle(request):
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"choices": [{"token_ids": [9, 8, 7], "text": "discarded"}], "usage": {}})
    async def run():
        config = SimpleNamespace(prompt_length=8192, response_length=24576,
                                 plugin=SimpleNamespace(turn_max_new_tokens=2048))
        tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "<think>reason</think>answer")
        client = TokenClient("http://local-model", "model", tokenizer, config, 42, tmp_path / "audit.jsonl")
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://local-model")
        try:
            result = await client.create_completion([1, 2, 3], structured_outputs={"json": {"type": "object"}})
            assert result["choices"][0]["message"]["raw_output_ids"] == [9, 8, 7]
            assert "reason" in result["choices"][0]["message"]["content"]
        finally:
            await client.client.aclose()
    asyncio.run(run())
    assert requests[0]["prompt"] == [1, 2, 3]
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["structured_outputs"] == {"json": {"type": "object"}}


def test_summary_rejects_missing_rows_and_judge_errors(tmp_path):
    manifest = dict(source_sha256="hash", indices=[0, 1, 2], commit="sha", model_path="snapshot",
                    method="foldagent", config={}, seed=42, judge_model="judge")
    for rank in range(3):
        (tmp_path / f"manifest-{rank}.json").write_text(json.dumps(manifest))
        row = dict(source_index=rank, task_reward=1, is_finish=True, status="ok")
        (tmp_path / f"results-{rank}.jsonl").write_text(json.dumps(row) + "\n")
    summarize(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["task_accuracy"] == 1
    (tmp_path / "results-2.jsonl").write_text("")
    with pytest.raises(ValueError, match="Missing"):
        summarize(tmp_path)
    row.update(env_stats={"judge_parse_failure": 1})
    (tmp_path / "results-2.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="judge"):
        summarize(tmp_path)
