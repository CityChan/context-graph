"""Opt-in real ScienceWorld JVM checks; no model API calls."""
import asyncio
import os
import subprocess
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
