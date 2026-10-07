import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from envs.discoveryworld_observation import decode, dumps, encode
from envs.discoveryworld_env import DiscoveryWorldEnv
from scripts.eval_agent_benchmarks import config_for
from scripts.audit_discoveryworld_observations import audit
from scripts.audit_scienceworld_pair import protocol_check
from tests.test_discoveryworld import FakeAPI, call


def snapshot():
    objects = [{"uuid": i, "name": "stone", "description": "A grey stone with markings that can be examined.",
                "distance": i % 3} for i in range(30)]
    return {"nearbyObjects": {"note": "Navigation only", "distance": 3, "objects": {"north": objects, "south": []}},
            "inventoryObjects": [{"uuid": 70, "name": "instrument", "description": "Result: 12.3 ± 0.1 °C"}],
            "accessibleEnvironmentObjects": [], "world_steps": 3,
            "taskProgress": [{"description": "Read all markings; compare measurements", "completed": False}],
            "dialog_box": {"dialogOptions": {"1": "Ask about dating"}},
            "extended_action_message": "Ancient inscription: αβγ", "new_field": {"text": 0}}


def test_round_trip_no_mutation_self_contained_and_size():
    ui = snapshot()
    before = deepcopy(ui)
    packed = encode(ui, "compact_v1")
    assert decode(json.loads(dumps(packed, "compact_v1"))) == before
    assert ui == before
    assert len(dumps(packed, "compact_v1")) < len(dumps(ui)) * .65
    changed = snapshot()
    changed["nearbyObjects"]["objects"]["north"][0]["description"] = "Changed measurement"
    assert decode(encode(changed, "compact_v1")) == changed
    assert decode(packed) == before
    assert dumps(encode(ui)) == json.dumps(ui, ensure_ascii=False)


@pytest.mark.parametrize("rows", [[], [{"uuid": 1}, {"uuid": 2, "name": None}],
    [{"uuid": 1, "extra": {"text": 0}}], [{"uuid": 1, "name": None}],
    [{"uuid": 1, "name": ""}, {"uuid": 1, "name": ""}]])
def test_unknown_nested_missing_null_and_duplicate_values_preserved(rows):
    ui = snapshot()
    ui["inventoryObjects"] = rows
    assert decode(json.loads(dumps(encode(ui, "compact_v1")))) == ui


@pytest.mark.parametrize("profile", ["full", "compact_v1"])
def test_action_feedback_and_raw_audit_preserved(profile, tmp_path):
    env = DiscoveryWorldEnv(SimpleNamespace(plugin=SimpleNamespace(discoveryworld_max_steps=2,
                             discoveryworld_observation_profile=profile)), None, "DiscoveryWorld@real")
    env._env = FakeAPI()
    ui = snapshot()
    env._env.ui = [SimpleNamespace(renderJSON=lambda: deepcopy(ui))]
    env._actions = {"MOVE_DIRECTION": {"args": ["arg1"]}}
    path = tmp_path / "tools.jsonl"
    env.instance_info = {"tool_log": str(path)}
    result = asyncio.run(env.run_action(call({"action": "MOVE_DIRECTION", "arg1": "north"})))
    logged = json.loads(path.read_text(encoding="utf8"))
    model = json.loads(result["observation"])
    assert model["action_result"] == logged["result"]
    assert (decode(model["ui"]) if profile == "compact_v1" else model["ui"]) == logged["observation"]
    assert env._env.ticks == 1
    env._env = None
    report = audit(tmp_path)
    assert report["round_trip_passed"] and report["records"][0]["snapshots"] == 1


def test_pair_protocol_rejects_mixed_profiles_and_scienceworld_unchanged():
    from omegaconf import OmegaConf
    manifests = []
    for method in ("contextgraph", "foldagent"):
        config = config_for("discoveryworld", method, 65536, prompt_profile="discoveryworld_v1", observation_profile="compact_v1")
        manifests.append({"method": method, "task_ids": ["same"], "config": OmegaConf.to_container(config)})
    assert protocol_check([manifests[0]], [manifests[1]])["matched"]
    manifests[1]["config"]["actor_rollout_ref"]["rollout"]["plugin"]["discoveryworld_observation_profile"] = "full"
    assert not protocol_check([manifests[0]], [manifests[1]])["matched"]
    sw = config_for("scienceworld", "contextgraph", 65536)
    assert "discoveryworld_observation_profile" not in sw.actor_rollout_ref.rollout.plugin
    with pytest.raises(ValueError, match="only supported"):
        config_for("scienceworld", "contextgraph", 65536, observation_profile="compact_v1")


def test_child_process_receives_profile(monkeypatch, tmp_path):
    from scripts import eval_agent_benchmarks as runner
    args = SimpleNamespace(output=tmp_path, benchmark="discoveryworld", shard_index=0, shard_count=1,
        retry_errors=False, method="contextgraph", endpoint="unused", model_path="unused",
        context_length=65536, max_steps=100, memory_profile="repaired", prompt_profile="discoveryworld_v1",
        discoveryworld_observation_profile="compact_v1", task_timeout=30)
    monkeypatch.setattr(runner, "preflight", lambda args: ([{"task_id": "test"}], {}))
    monkeypatch.setattr(runner, "summary", lambda *args: {"infrastructure_errors": 0})
    calls = []
    def run_command(command, log, timeout):
        calls.append(command)
        (log.parent / "result.json").write_text(json.dumps({"status": "graded", "score": 0, "success": False}))
    monkeypatch.setattr(runner, "run_command", run_command)
    runner.run(args)
    assert calls[0][calls[0].index("--discoveryworld-observation-profile") + 1] == "compact_v1"
