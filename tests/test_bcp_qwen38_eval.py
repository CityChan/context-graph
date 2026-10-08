import asyncio
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts.eval_bcp_qwen38 import TokenClient, completion_budget, select_indices, summarize


def test_gaia_data_rejects_wrong_benchmark_attachments_and_duplicate_ids():
    from scripts.eval_bcp_qwen38 import validate_dataset
    from scripts.make_gaia_data import _row_to_task
    row = _row_to_task({"Question": "Test?", "Final answer": "42", "task_id": "a"}, 0, "search")
    validate_dataset([row], "gaia")
    with pytest.raises(ValueError, match="unique"):
        validate_dataset([row, row], "gaia")
    row["extra_info"]["file_name"] = "attachment.pdf"
    with pytest.raises(ValueError, match="attachments"):
        validate_dataset([row], "gaia")
    row["extra_info"]["file_name"] = ""
    row["extra_info"]["answer"] = ""
    with pytest.raises(ValueError, match="reference answers"):
        validate_dataset([row], "gaia")
    row["ability"] = "LocalSearch"
    with pytest.raises(ValueError, match="prepared GAIA"):
        validate_dataset([row], "gaia")


@pytest.mark.parametrize("method,executor", [
    ("react", "react_agent.py"), ("foldagent", "fold_agent.py"),
    ("contextgraph", "graph_agent_isolated.py")])
def test_eval_config_satisfies_executor_root_config_reads(method, executor):
    # Check the real executor's config dependencies without importing GPU/Ray
    # modules or contacting a model. This catches absent parents of getattr.
    from scripts.eval_bcp_qwen38 import config_for
    config = config_for(SimpleNamespace(method=method, memory_mode="repaired"))
    source = Path(__file__).resolve().parents[1] / "agents" / executor
    reads = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute):
            expression = ast.unparse(node)
            if expression.startswith("context.config."):
                reads.append(expression)
                value = config
                for key in expression.split(".")[2:]:
                    value = getattr(value, key)
    assert reads
    assert config.algorithm.adv_estimator not in {"graphrpo", "AdvantageEstimator.GRAPHRPO"}
    assert config.actor_rollout_ref.rollout.plugin.process_reward is None


def test_react_uses_linear_workflow_with_matched_evaluation_budget():
    from scripts.eval_bcp_qwen38 import config_for
    react = config_for(SimpleNamespace(method="react", memory_mode="repaired"))
    fold = config_for(SimpleNamespace(method="foldagent", memory_mode="repaired"))
    rollout = react.actor_rollout_ref.rollout
    other = fold.actor_rollout_ref.rollout
    assert rollout.plugin.workflow == "search"
    assert not rollout.plugin.structured_graph_controller
    assert not rollout.plugin.controller_owned_tool_formatting
    assert not rollout.plugin.structured_memory_enabled
    assert rollout.prompt_length == other.prompt_length == 8192
    assert rollout.response_length == other.response_length == 24576
    for key in ("final_answer_reserve", "turn_max_new_tokens", "search_topk_cap",
                "max_turn", "apply_chat_template_kwargs"):
        assert rollout.plugin[key] == other.plugin[key]


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


@pytest.mark.parametrize("model", ["Qwen/Qwen3.8-27B", "Qwen/Qwen3.5-9B"])
@pytest.mark.parametrize("constraint", [{"json": {"type": "object"}}, {"regex": "<summary>[^<>]{1,928}</summary>"}])
def test_token_client_keeps_exact_ids_and_uses_no_judge_credentials(tmp_path, monkeypatch, model, constraint):
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
        client = TokenClient("http://local-model", model, tokenizer, config, 42, tmp_path / "audit.jsonl")
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://local-model")
        try:
            result = await client.create_completion([1, 2, 3], structured_outputs=constraint)
            assert result["choices"][0]["message"]["raw_output_ids"] == [9, 8, 7]
            assert "reason" in result["choices"][0]["message"]["content"]
        finally:
            await client.client.aclose()
    asyncio.run(run())
    assert requests[0]["prompt"] == [1, 2, 3]
    assert requests[0]["model"] == model
    assert requests[0]["return_token_ids"] is True
    assert requests[0]["structured_outputs"] == constraint
    assert json.loads((tmp_path / "audit.jsonl").read_text())["structured_outputs"] == constraint


def test_summary_rejects_missing_rows_and_judge_errors(tmp_path):
    manifest = dict(source_sha256="hash", indices=[0, 1, 2], commit="sha", model_path="snapshot",
                    method="foldagent", config={}, seed=42, judge_model="judge")
    for rank in range(3):
        (tmp_path / f"manifest-{rank}.json").write_text(json.dumps(manifest))
        row = dict(source_index=rank, task_reward=1, is_finish=True, status="ok")
        (tmp_path / f"results-{rank}.jsonl").write_text(json.dumps(row) + "\n")
    summarize(tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["task_accuracy"] == 1
    manifest["model"] = "Qwen/Qwen3.5-9B"
    (tmp_path / "manifest-2.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model mismatch"):
        summarize(tmp_path)
    manifest.pop("model")
    (tmp_path / "manifest-2.json").write_text(json.dumps(manifest))
    (tmp_path / "results-2.jsonl").write_text("")
    with pytest.raises(ValueError, match="Missing"):
        summarize(tmp_path)
    row.update(env_stats={"judge_parse_failure": 1})
    (tmp_path / "results-2.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(RuntimeError, match="judge"):
        summarize(tmp_path)
