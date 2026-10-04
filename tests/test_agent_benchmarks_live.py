"""Opt-in real JVM / upstream-metric checks; never make paid model/search calls."""
import asyncio
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(not os.environ.get("SCIENCEWORLD_LIVE_TEST"), reason="requires local Java and scienceworld==1.2.3")
def test_live_scienceworld_test_split_and_episode(monkeypatch, tmp_path):
    from scripts.prepare_agent_benchmarks import prepare
    from scripts.eval_agent_benchmarks import load_tasks
    from envs.scienceworld_env import ScienceWorldEnv
    from tests.test_agent_benchmarks import item
    if os.name == "nt":
        original = subprocess.Popen
        def hidden(*args, **kwargs):
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            return original(*args, **kwargs)
        monkeypatch.setattr(subprocess, "Popen", hidden)
    prepare("scienceworld", tmp_path)
    source, tasks = load_tasks(tmp_path / "tasks.json", "scienceworld", -1)
    assert source["split"] == "test" and len({t["task_name"] for t in tasks}) == 30
    assert all(t["split"] == "test" for t in tasks)
    async def episode():
        env = ScienceWorldEnv(SimpleNamespace(plugin=SimpleNamespace(scienceworld_max_steps=2)), None, "ScienceWorld@real")
        try:
            await env.init_env(item(tasks[0]))
            for _ in range(2):
                result = await env.run_action("<function=action><parameter=command>look around</parameter></function>")
            assert result["action"] == "finish"
            assert env.stats["environment_steps"] == 2
            assert env.stats["completed"] == 0
            assert (await env.get_reward(None, [], None))[1] == 0
        finally:
            env.close()
    asyncio.run(episode())


@pytest.mark.skipif(not os.environ.get("WIDESEARCH_TEST_UPSTREAM"), reason="requires pinned upstream checkout")
@pytest.mark.parametrize("mode", ["perfect", "partial", "bad_judge", "malformed_table", "provider_error"])
def test_official_widesearch_metrics(monkeypatch, tmp_path, mode):
    from scripts.grade_widesearch import grade
    from scripts.prepare_agent_benchmarks import verify_evaluator
    from scripts.eval_discoverybench_qwen35 import save
    import openai
    upstream = Path(os.environ["WIDESEARCH_TEST_UPSTREAM"])
    verify_evaluator(upstream)
    monkeypatch.setattr("scripts.grade_widesearch.verify_evaluator", lambda _: verify_evaluator(upstream))
    monkeypatch.syspath_prepend(str(upstream))
    # Isolate upstream src.* modules between tests.
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            monkeypatch.delitem(sys.modules, name)
    class Client:
        def __init__(self, **kwargs): self.chat = SimpleNamespace(completions=self)
        def create(self, **kwargs):
            if mode == "provider_error": raise TimeoutError("judge unavailable")
            content = "not JSON" if mode == "bad_judge" else '```json\n{"idx_0": 1}\n```'
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], model="fake", usage=None)
        def close(self): pass
    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    reference = {"evaluation": {"required": ["name", "value"], "unique_columns": ["name"],
                  "eval_pipeline": {"value": {"metric": ["llm_judge"], "criterion": "same value"}}},
                  "csv": "name,value\nA,1\n" + ("B,2\n" if mode == "partial" else "")}
    save(tmp_path / "references.json", {"test": reference})
    (tmp_path / "prediction.md").write_text("no table" if mode == "malformed_table" else "| name | value |\n| --- | --- |\n| A | 1 |")
    task = {"task_id": "test", "query": "q", "language": "en"}
    if mode in ("bad_judge", "provider_error"):
        with pytest.raises(RuntimeError, match="evaluator failed"):
            grade(tmp_path / "tasks.json", task, tmp_path / "prediction.md", tmp_path / "result.json", "fake")
        assert not (tmp_path / "result.json").exists()
    else:
        grade(tmp_path / "tasks.json", task, tmp_path / "prediction.md", tmp_path / "result.json", "fake")
        record = json.loads((tmp_path / "result.json").read_text())
        assert record["score"] == int(mode == "perfect")
        if mode == "partial": assert record["metrics"]["f1_by_row"] == pytest.approx(2/3)
