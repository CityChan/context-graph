import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from envs.discoveryworld_env import DiscoveryWorldEnv
from envs.discoveryworld_protocol import SCENARIOS, tasks_for, verify_install
from scripts.eval_agent_benchmarks import config_for, load_tasks
from scripts.prepare_agent_benchmarks import prepare


def call(command):
    return '<function=action><parameter=command>' + json.dumps(command) + '</parameter></function>'


def item(extra):
    import numpy as np
    return SimpleNamespace(non_tensor_batch={"extra_info": np.array([extra], dtype=object)})


def test_public_suite_and_selection(tmp_path):
    prepare("discoveryworld", tmp_path)
    source, tasks = load_tasks(tmp_path / "tasks.json", "discoveryworld", -1)
    assert source["split"] == "public"
    assert len(tasks) == 40 and len(tasks_for("all")) == 120
    assert {t["scenario"] for t in tasks[:8]} == set(SCENARIOS)
    assert {t["seed"] for t in tasks} == set(range(5))
    bundle = json.loads((tmp_path / "tasks.json").read_text())
    bundle["tasks"][0]["answer"] = "secret"
    (tmp_path / "tasks.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="catalogue"):
        load_tasks(tmp_path / "tasks.json", "discoveryworld", -1)


@pytest.mark.parametrize("method", ["contextgraph", "foldagent"])
def test_paired_config_prompt_dispatch(method):
    from agents.prompts import create_chat
    from agents.utils import select_env
    plugin = config_for("discoveryworld", method, 65536, 200, prompt_profile="discoveryworld_v1").actor_rollout_ref.rollout.plugin
    assert plugin.max_turn == plugin.val_max_turn == plugin.discoveryworld_max_steps == 200
    assert not plugin.graph_controller_counts_as_turn
    assert select_env("DiscoveryWorld@real", None) is DiscoveryWorldEnv
    chat = create_chat("experiment", plugin.workflow, expose_graph_tools=False)
    assert "chosen_dialog_option_int" in chat[0]["content"]
    assert "ScienceWorld" not in chat[0]["content"]


class FakeAPI:
    def __init__(self):
        self.ticks = 0
        self.dialog = False
        self.calls = []
        self.ui = [SimpleNamespace(renderJSON=lambda: {"world_steps": self.ticks,
             "lastActionMessage": "measurement", "taskProgress": [{"description": "test", "score": "SECRET"}]})]

    def isAgentInDialog(self, index):
        return self.dialog

    def performAgentAction(self, index, command):
        self.calls.append(command)
        return {"success": False, "errors": ["cannot move"]}

    def tick(self):
        self.ticks += 1
        return {"success": True}

    def getTaskScorecard(self):
        return [{"scoreNormalized": .25, "completed": False, "completedSuccessfully": False,
                 "criticalHypotheses": ["SECRET"]}]


def test_ticks_invalid_dialog_score_isolation(tmp_path):
    env = DiscoveryWorldEnv(SimpleNamespace(plugin=SimpleNamespace(discoveryworld_max_steps=2)), None, "DiscoveryWorld@real")
    api = FakeAPI()
    env._env = api
    env._actions = {"MOVE_DIRECTION": {"args": ["arg1"]}}
    env.instance_info = {"grading_log": str(tmp_path / "scorecard.json")}
    try:
        bad = asyncio.run(env.run_action(call({"action": "MOVE_DIRECTION", "arg1": []})))
        assert api.ticks == 0 and env.stats["invalid_tool"] == 1
        result = asyncio.run(env.run_action(call({"action": "MOVE_DIRECTION", "arg1": "north"})))
        assert api.ticks == 1 and len(api.calls) == 1
        assert "SECRET" not in result["observation"]
        assert "SECRET" in (tmp_path / "scorecard.json").read_text()
        assert env.stats["environment_score"] == 25
        api.dialog = True
        result = asyncio.run(env.run_action(call({"chosen_dialog_option_int": 1})))
        assert api.ticks == 2 and result["action"] == "finish"
        asyncio.run(env.run_action(call({"chosen_dialog_option_int": 1})))
        assert api.ticks == 2
    finally:
        env._env = None  # FakeAPI owns no pygame resources.


@pytest.mark.parametrize("success", [False, True])
def test_terminal_is_not_necessarily_success(success):
    env = DiscoveryWorldEnv(SimpleNamespace(plugin=SimpleNamespace(discoveryworld_max_steps=10)), None, "DiscoveryWorld@real")
    env._env = FakeAPI()
    env._actions = {"MOVE_DIRECTION": {"args": ["arg1"]}}
    env._env.getTaskScorecard = lambda: [{"scoreNormalized": .5, "completed": True, "completedSuccessfully": success}]
    try:
        result = asyncio.run(env.run_action(call({"action": "MOVE_DIRECTION", "arg1": "north"})))
        assert result["action"] == "finish"
        assert asyncio.run(env.get_reward(None, [], None))[1] == float(success)
        assert env.stats["environment_step_limit"] == 0
    finally:
        env._env = None


def test_pair_audit_requires_matching_protocol_and_common_grades(tmp_path):
    from omegaconf import OmegaConf
    from scripts.audit_discoveryworld_pair import audit
    from scripts.eval_discoverybench_qwen35 import save, task_key
    ids = [t["task_id"] for t in tasks_for()[:2]]
    for method in ("contextgraph", "foldagent"):
        root = tmp_path / method
        for identity in ids:
            (root / "instances" / task_key(identity)).mkdir(parents=True)
        manifest = {"method": method, "task_ids": ids, "source": {"benchmark": "discoveryworld"},
                    "model": "same", "config": OmegaConf.to_container(config_for("discoveryworld", method, 65536, prompt_profile="discoveryworld_v1"))}
        save(root / "manifest.json", manifest)
        save(root / "instances" / task_key(ids[0]) / "result.json", {"task_id": ids[0], "attempt": "/remote/attempt-1",
             "status": "graded", "score": 25 if method == "contextgraph" else 0, "success": False})
        save(root / "instances" / task_key(ids[1]) / "result.json", {"task_id": ids[1], "attempt": "/remote/attempt-2",
             "status": "infrastructure_error"})
    report = audit(tmp_path, tmp_path)
    assert report["protocol"]["matched"]
    assert report["common"]["count"] == 1
    assert report["common"]["score_delta_contextgraph_minus_foldagent"] == 25
    manifest["model"] = "different"
    save(tmp_path / "foldagent" / "manifest.json", manifest)
    assert not audit(tmp_path, tmp_path)["protocol"]["matched"]


def test_simulator_errors_are_not_agent_failures():
    env = DiscoveryWorldEnv(SimpleNamespace(plugin=SimpleNamespace(discoveryworld_max_steps=10)), None, "DiscoveryWorld@real")
    env._env = FakeAPI()
    env._actions = {"MOVE_DIRECTION": {"args": ["arg1"]}}
    env._env.tick = lambda: {"success": False}
    try:
        with pytest.raises(RuntimeError, match="tick"):
            asyncio.run(env.run_action(call({"action": "MOVE_DIRECTION", "arg1": "north"})))
        assert env.env_fail and env.stats["env_error"] == 1
    finally:
        env._env = None


@pytest.mark.skipif(not os.environ.get("DISCOVERYWORLD_LIVE_TEST"), reason="opt-in real pinned simulator")
@pytest.mark.parametrize("method", ["contextgraph", "foldagent"])
@pytest.mark.parametrize("observation_profile", ["full", "compact_v1"])
def test_real_agent_loop_and_simulator(method, observation_profile, tmp_path):
    import importlib
    import numpy as np
    from tests.test_session_restart import Tokenizer, Client
    from verl import DataProto
    module = importlib.import_module("agents.graph_agent_isolated" if method == "contextgraph" else "agents.fold_agent")
    config = config_for("discoveryworld", method, 65536, 2, prompt_profile="discoveryworld_v1", observation_profile=observation_profile)
    extra = dict(tasks_for()[0], workflow=config.actor_rollout_ref.rollout.plugin.workflow,
                 grading_log=str(tmp_path / "scorecard.json"))
    task = DataProto()
    task.non_tensor_batch = {"ability": np.array(["DiscoveryWorld@real"], dtype=object),
        "extra_info": np.array([extra], dtype=object), "uid": np.array([extra["task_id"]], dtype=object)}
    task.meta_info = {"generation_kwargs": {}, "max_turn": 2}
    client = Client([call({"action": "ROTATE_DIRECTION", "arg1": "north"})] * 2)
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), llm_client=client, is_train=False, global_step=0)
    out = asyncio.run(module.process_item(task, context))
    assert len(client.calls) == 2
    for ids, _ in client.calls:
        assert ("discoveryworld.compact.v1" in Tokenizer().decode(ids)) == (observation_profile == "compact_v1")
    assert out[0].extra_fields["env_stats"]["environment_steps"] == 2
    for _, kwargs in client.calls:
        assert "criticalHypotheses" not in str(kwargs.get("messages"))
        assert "scoreNormalized" not in str(kwargs.get("messages"))


@pytest.mark.skipif(not os.environ.get("DISCOVERYWORLD_LIVE_TEST"), reason="opt-in real pinned simulator")
@pytest.mark.parametrize("task", tasks_for()[:8], ids=[t["scenario"] for t in tasks_for()[:8]])
def test_real_scenarios(task, tmp_path):
    verify_install()
    async def episode():
        env = DiscoveryWorldEnv(SimpleNamespace(plugin=SimpleNamespace(discoveryworld_max_steps=2)), None, "DiscoveryWorld@real")
        try:
            await env.init_env(item(dict(task, tool_log=str(tmp_path / "tools.jsonl"), grading_log=str(tmp_path / "scorecard.json"))))
            assert "criticalHypotheses" not in env.instance_info["problem_statement"]
            assert "criticalQuestions" not in env.instance_info["problem_statement"]
            start = env._env.steps
            for _ in range(2):
                result = await env.run_action(call({"action": "ROTATE_DIRECTION", "arg1": "north"}))
            assert env._env.steps == start + 2
            assert result["action"] == "finish" and env.stats["environment_steps"] == 2
            assert not env.env_fail
            assert 0 <= env.stats["environment_score"] <= 100
        finally:
            env.close()
    asyncio.run(episode())
