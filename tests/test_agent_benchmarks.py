import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from envs.scienceworld_env import ScienceWorldEnv
from envs.widesearch_env import WideSearchEnv
from scripts.eval_agent_benchmarks import config_for, load_tasks, run_command, summary
from scripts.eval_discoverybench_qwen35 import save, task_key


def item(extra):
    return SimpleNamespace(non_tensor_batch={"extra_info": np.array([extra], dtype=object)})


@pytest.mark.parametrize("score,done,success", [(100, True, True), (40, True, False), (-10, True, False), (40, False, False)])
def test_scienceworld_done_is_not_success(monkeypatch, tmp_path, score, done, success):
    class Simulator:
        def __init__(self, **kwargs):
            self.closed = False
        def load(self, **kwargs):
            assert kwargs["generateGoldPath"] is False
        def reset(self):
            return "room", {"score": 0, "valid": ["look around"]}
        def get_task_description(self):
            return "goal"
        def step(self, command):
            return "observation", 1, done, {"score": score, "valid": [], "moves": 1}
        def close(self):
            self.closed = True
    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=Simulator))
    config = SimpleNamespace(plugin=SimpleNamespace(scienceworld_max_steps=1))
    env = ScienceWorldEnv(config, None, "ScienceWorld@real")
    task = item({"task_name": "test", "tool_log": str(tmp_path / "tools.jsonl")})
    asyncio.run(env.init_env(task))
    asyncio.run(env.run_action("<function=action><parameter=command>look around</parameter></function>"))
    assert env.is_finish  # either simulator terminal or adapter action cap
    assert env.stats["environment_score"] == score
    assert env.stats["invalid_actions"] == 0  # empty valid-action list is not an invalid-action flag
    assert asyncio.run(env.get_reward(task, [], None))[1] == int(success)
    assert len((tmp_path / "tools.jsonl").read_text().splitlines()) == 2
    sim = env._env
    env.close()
    assert sim.closed


def test_scienceworld_error_closes_simulator():
    class Broken:
        closed = False
        def step(self, command):
            raise OSError("JVM exited")
        def close(self):
            self.closed = True
    env = ScienceWorldEnv(SimpleNamespace(plugin=None), None, "ScienceWorld@real")
    sim = env._env = Broken()
    with pytest.raises(RuntimeError, match="simulator"):
        asyncio.run(env.run_action("<function=action><parameter=command>look around</parameter></function>"))
    assert env.env_fail and env.is_finish and sim.closed


def test_widesearch_tools_preserve_table_and_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("TAVILY_API_KEY", "test-secret")
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/search":
            assert body["max_results"] == 20
            return httpx.Response(200, json={"results": [{"url": "https://example.org", "content": "fact"}]})
        return httpx.Response(200, json={"results": [{"raw_content": "x" * 30}]})
    async def scenario():
        env = WideSearchEnv(SimpleNamespace(plugin=SimpleNamespace(widesearch_page_chars=10)), None, "WideSearch")
        await env.init_env(item({"query": "Find rows", "tool_log": str(tmp_path / "tools.jsonl"),
                                 "prediction_path": str(tmp_path / "prediction.md")}))
        await env.client.aclose()
        env.client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://api.tavily.com")
        obs = await env.run_action("<function=search><parameter=query>test</parameter><parameter=topk>99</parameter></function>")
        assert json.loads(obs["observation"])[0]["docid"] == "1"
        obs = await env.run_action("<function=open_page><parameter=docid>1</parameter></function>")
        assert obs["observation"].endswith("x" * 10)
        table = "| Name |\n| --- |\n| Example |"
        await env.run_action(f"<function=finish><parameter=answer>{table}</parameter></function>")
        assert (tmp_path / "prediction.md").read_text() == table
        assert env.is_finish and env.stats["output_truncations"] == 1
        await env.aclose()
    asyncio.run(scenario())
    assert calls[1][1]["urls"] == ["https://example.org"]
    assert "test-secret" not in (tmp_path / "tools.jsonl").read_text()


def test_provider_failure_is_not_a_zero_score(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    async def scenario():
        env = WideSearchEnv(SimpleNamespace(plugin=None), None, "WideSearch")
        await env.init_env(item({"query": "test"}))
        await env.client.aclose()
        env.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(429)), base_url="https://api.tavily.com")
        try:
            with pytest.raises(RuntimeError, match="provider"):
                await env.run_action("<function=search><parameter=query>test</parameter></function>")
            assert env.env_fail and env.stats["provider_errors"] == 1
        finally:
            await env.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["answer", "evaluation", "gold", "gold_path"])
def test_gold_never_enters_web_environment(field):
    env = WideSearchEnv(SimpleNamespace(plugin=None), None, "WideSearch")
    with pytest.raises(ValueError, match="grading"):
        asyncio.run(env.init_env(item({"query": "test", field: "secret"})))


def test_dataset_rejects_gold_and_wrong_split(tmp_path):
    path = tmp_path / "tasks.json"
    save(path, {"source": {"benchmark": "widesearch"}, "tasks": [{"task_id": "x", "query": "q", "language": "en", "answer": "leak"}]})
    with pytest.raises(ValueError, match="references"):
        load_tasks(path, "widesearch", -1)
    save(path, {"source": {"benchmark": "scienceworld"}, "tasks": [{"task_id": "x", "task_name": "q", "variation_idx": 0, "split": "train"}]})
    with pytest.raises(ValueError, match="test split"):
        load_tasks(path, "scienceworld", -1)


def test_summary_does_not_count_infrastructure_as_agent_failure(tmp_path):
    for identity, result in [("a", {"status": "graded", "score": 40, "success": False}),
                             ("b", {"status": "infrastructure_error"})]:
        path = tmp_path / "instances" / task_key(identity)
        path.mkdir(parents=True)
        save(path / "result.json", result)
    value = summary(tmp_path, ["a", "b", "c"], "scienceworld")
    assert value["graded"] == 1 and value["infrastructure_errors"] == 1 and value["pending"] == 1
    assert value["mean_score_graded"] == 40 and value["mean_score"] is None


@pytest.mark.parametrize("benchmark", ["scienceworld", "widesearch"])
def test_paired_protocol_matches_model_budgets(benchmark):
    cg = config_for(benchmark, "contextgraph", 65536).actor_rollout_ref.rollout
    fa = config_for(benchmark, "foldagent", 65536).actor_rollout_ref.rollout
    assert cg.response_length == fa.response_length == 57344
    assert cg.plugin.final_answer_reserve == fa.plugin.final_answer_reserve
    assert cg.plugin.max_session == fa.plugin.max_session
    assert cg.plugin.structured_graph_controller and not fa.plugin.structured_graph_controller


def test_widesearch_dispatch_and_table_prompt():
    from agents.utils import select_env
    from agents.prompts import create_chat
    assert select_env("WideSearch", None) is WideSearchEnv
    for method in ("branch", "graph"):
        chat = create_chat("List cities", "widesearch_" + method)
        assert "complete Markdown table" in chat[0]["content"]


def test_subprocess_timeout_and_nonzero_are_errors(tmp_path):
    script = tmp_path / "child.py"
    script.write_text("import time\ntime.sleep(60)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, str(script)], tmp_path / "log.txt", 0.1)
    script.write_text("raise SystemExit(7)\n")
    with pytest.raises(RuntimeError, match="exit 7"):
        run_command([sys.executable, str(script)], tmp_path / "log.txt", 10)


@pytest.mark.parametrize("method", ["contextgraph", "foldagent"])
@pytest.mark.parametrize("benchmark", ["scienceworld", "widesearch"])
def test_real_agent_loop_stops_on_environment_finish(monkeypatch, tmp_path, method, benchmark):
    import importlib
    from tests.test_session_restart import Tokenizer, Client
    from verl import DataProto
    module = importlib.import_module("agents.graph_agent_isolated" if method == "contextgraph" else "agents.fold_agent")
    config = config_for(benchmark, method, 65536)
    extra = {"workflow": config.actor_rollout_ref.rollout.plugin.workflow}
    if benchmark == "scienceworld":
        class Simulator:
            def __init__(self, **kwargs): pass
            def load(self, **kwargs): pass
            def reset(self): return "room", {"score": 0}
            def get_task_description(self): return "goal"
            def step(self, command): return "done", 100, True, {"score": 100, "valid": []}
            def close(self): pass
        monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=Simulator))
        extra["task_name"] = "test"
        response = "<function=action><parameter=command>look around</parameter></function>"
        ability = "ScienceWorld@real"
    else:
        monkeypatch.setenv("TAVILY_API_KEY", "test")
        extra.update(query="List places", prediction_path=str(tmp_path / "prediction.md"))
        response = "<function=finish><parameter=answer>| Name |\n| --- |\n| London |</parameter></function>"
        ability = "WideSearch"
    task = DataProto()
    task.non_tensor_batch = {"ability": np.array([ability], dtype=object), "extra_info": np.array([extra], dtype=object),
                             "uid": np.array(["test"], dtype=object)}
    task.meta_info = {"generation_kwargs": {}}
    client = Client([response])
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), llm_client=client, is_train=False, global_step=0)
    out = asyncio.run(module.process_item(task, context))
    assert len(client.calls) == 1
    assert out[0].extra_fields["is_finish"]
    if benchmark == "scienceworld":
        assert out[0].extra_fields["env_stats"]["environment_score"] == 100
    else:
        assert "London" in (tmp_path / "prediction.md").read_text()


def test_resume_skips_completed_and_retries_only_errors(monkeypatch, tmp_path):
    import scripts.eval_agent_benchmarks as runner
    tasks = [{"task_id": "a"}, {"task_id": "b"}]
    monkeypatch.setattr(runner, "preflight", lambda _: (tasks, {"revision": "one"}))
    calls = []
    def command(argv, log, timeout):
        attempt = Path(argv[argv.index("--task") + 1])
        identity = json.loads((attempt / "task.json").read_text())["task_id"]
        calls.append(identity)
        if identity == "b" and calls.count("b") == 1:
            raise RuntimeError("JVM failed")
        save(attempt / "result.json", {"status": "graded", "score": 100, "success": True})
    monkeypatch.setattr(runner, "run_command", command)
    args = SimpleNamespace(output=tmp_path, benchmark="scienceworld", method="foldagent", endpoint="unused",
                           model_path="unused", context_length=65536, max_steps=100, task_timeout=30,
                           shard_index=0, shard_count=1, retry_errors=False)
    assert runner.run(args) == 2
    assert runner.run(args) == 2
    assert calls == ["a", "b"]
    args.retry_errors = True
    assert runner.run(args) == 0
    assert calls == ["a", "b", "b"]
    assert len(list((tmp_path / "instances" / task_key("b")).glob("attempt-*"))) == 2
    monkeypatch.setattr(runner, "preflight", lambda _: (tasks, {"revision": "changed"}))
    with pytest.raises(ValueError, match="Resume protocol"):
        runner.run(args)


def test_judge_retry_reuses_prediction_without_regenerating(monkeypatch, tmp_path):
    import scripts.eval_agent_benchmarks as runner
    monkeypatch.setattr(runner, "preflight", lambda _: ([{"task_id": "a"}], {"revision": "one"}))
    calls = []
    def command(argv, log, timeout):
        if "--prediction" not in argv:
            calls.append("generate")
            attempt = Path(argv[argv.index("--task") + 1])
            (attempt / "prediction.md").write_text("fixed answer")
            save(attempt / "trajectory.json", [])
            return
        calls.append("grade")
        assert Path(argv[argv.index("--prediction") + 1]).read_text() == "fixed answer"
        if calls.count("grade") == 1:
            raise RuntimeError("Judge unavailable")
        metrics = {k: 1.0 for k in ("f1_by_row", "f1_by_item", "precision_by_row", "recall_by_row", "precision_by_item", "recall_by_item")}
        save(Path(argv[argv.index("--output") + 1]), {"status": "graded", "score": 1,
                                                   "valid_prediction": True, "metrics": metrics})
    monkeypatch.setattr(runner, "run_command", command)
    args = SimpleNamespace(output=tmp_path, benchmark="widesearch", method="foldagent", endpoint="unused",
                           model_path="unused", context_length=65536, max_steps=100, task_timeout=30,
                           grade_timeout=30, data=tmp_path / "tasks.json", judge_model="fake",
                           shard_index=0, shard_count=1, retry_errors=True)
    assert runner.run(args) == 2
    assert runner.run(args) == 0
    assert calls == ["generate", "grade", "grade"]
    result = json.loads((tmp_path / "instances" / task_key("a") / "result.json").read_text())
    assert result["generation_reused_from"]
