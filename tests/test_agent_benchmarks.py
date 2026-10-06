import asyncio
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from envs.scienceworld_env import ScienceWorldEnv
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


def test_dataset_rejects_gold_and_wrong_split(tmp_path):
    path = tmp_path / "tasks.json"
    save(path, {"source": {"benchmark": "scienceworld"}, "tasks": [{"task_id": "x", "task_name": "q", "variation_idx": 0, "split": "test", "answer": "leak"}]})
    with pytest.raises(ValueError, match="references"):
        load_tasks(path, "scienceworld", -1)
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


def test_paired_protocol_matches_model_budgets():
    benchmark = "scienceworld"
    cg = config_for(benchmark, "contextgraph", 65536).actor_rollout_ref.rollout
    fa = config_for(benchmark, "foldagent", 65536).actor_rollout_ref.rollout
    assert cg.response_length == fa.response_length == 57344
    assert cg.plugin.final_answer_reserve == fa.plugin.final_answer_reserve
    assert cg.plugin.max_session == fa.plugin.max_session
    assert cg.plugin.graph_controller_counts_as_turn is False
    assert fa.plugin.graph_controller_counts_as_turn is False
    assert cg.plugin.structured_graph_controller and not fa.plugin.structured_graph_controller


def test_subprocess_timeout_and_nonzero_are_errors(tmp_path):
    script = tmp_path / "child.py"
    script.write_text("import time\ntime.sleep(60)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, str(script)], tmp_path / "log.txt", 0.1)
    script.write_text("raise SystemExit(7)\n")
    with pytest.raises(RuntimeError, match="exit 7"):
        run_command([sys.executable, str(script)], tmp_path / "log.txt", 10)


@pytest.mark.parametrize("method", ["contextgraph", "foldagent"])
def test_real_agent_loop_stops_on_environment_finish(monkeypatch, method):
    benchmark = "scienceworld"
    import importlib
    from tests.test_session_restart import Tokenizer, Client
    from verl import DataProto
    module = importlib.import_module("agents.graph_agent_isolated" if method == "contextgraph" else "agents.fold_agent")
    config = config_for(benchmark, method, 65536)
    extra = {"workflow": config.actor_rollout_ref.rollout.plugin.workflow}
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
    task = DataProto()
    task.non_tensor_batch = {"ability": np.array([ability], dtype=object), "extra_info": np.array([extra], dtype=object),
                             "uid": np.array(["test"], dtype=object)}
    task.meta_info = {"generation_kwargs": {}}
    client = Client([response])
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), llm_client=client, is_train=False, global_step=0)
    out = asyncio.run(module.process_item(task, context))
    assert len(client.calls) == 1
    assert out[0].extra_fields["is_finish"]
    assert out[0].extra_fields["env_stats"]["environment_score"] == 100


def test_resume_skips_completed_and_retries_only_errors(monkeypatch, tmp_path):
    import scripts.eval_agent_benchmarks as runner
    tasks = [{"task_id": "a"}, {"task_id": "b"}]
    monkeypatch.setattr(runner, "preflight", lambda _: (tasks, {"revision": "one"}))
    calls = []
    def command(argv, log, timeout):
        assert argv[argv.index("--memory-profile") + 1] == "repaired"
        attempt = Path(argv[argv.index("--task") + 1])
        identity = json.loads((attempt / "task.json").read_text())["task_id"]
        calls.append(identity)
        if identity == "b" and calls.count("b") == 1:
            raise RuntimeError("JVM failed")
        save(attempt / "result.json", {"status": "graded", "score": 100, "success": True})
    monkeypatch.setattr(runner, "run_command", command)
    args = SimpleNamespace(output=tmp_path, benchmark="scienceworld", method="foldagent", endpoint="unused",
                           model_path="unused", context_length=65536, max_steps=100, task_timeout=30,
                           shard_index=0, shard_count=1, retry_errors=False, memory_profile="repaired")
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
