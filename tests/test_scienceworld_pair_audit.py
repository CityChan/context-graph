import asyncio
import importlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_scienceworld_pair import audit, request_stats


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, method, scores, commit="same"):
    shard = root / (method + "-0")
    write(shard / "manifest.json", {"method": method, "task_ids": list(scores), "commit": commit,
                                    "model_revision": "same", "config": {}})
    for identity, score in scores.items():
        directory = shard / "instances" / identity
        attempt = directory / "attempt-current"
        write(attempt / "task.json", {"task_id": identity})
        write(directory / "result.json", {"task_id": identity, "attempt": str(attempt), "status": "graded",
                                           "score": score, "success": score == 100, "env_stats": {}})
    return root


def test_common_tasks_remove_completion_mix_and_preserve_ties(tmp_path):
    c = fixture(tmp_path / "c", "contextgraph", {"scienceworld_test_heat_1": 100, "scienceworld_test_heat_2": 0})
    f = fixture(tmp_path / "f", "foldagent", {"scienceworld_test_heat_1": 50, "scienceworld_test_heat_2": 0,
                                             "scienceworld_test_cool_3": 100})
    report = audit(c, f)
    assert report["common"]["count"] == 2
    assert report["common"]["score_delta_contextgraph_minus_foldagent"] == 25
    assert report["groups"]["equal"]["count"] == 1
    assert report["common"]["contextgraph_only_success"] == 1
    assert report["by_task_type"]["heat"]["count"] == 2
    assert report["protocol"]["matched"] is False  # selection differs
    assert report["common"]["contextgraph"]["metrics"]["consol_attempts"]["mean"] is None
    assert report["cases"]["scienceworld_test_heat_1"]["contextgraph"]["tool_log_available"] is False


def test_protocol_mismatch_and_current_attempt_only(tmp_path):
    scores = {"scienceworld_test_heat_1": 20}
    c = fixture(tmp_path / "c", "contextgraph", scores)
    f = fixture(tmp_path / "f", "foldagent", scores, commit="different")
    write(c / "contextgraph-0/instances/scienceworld_test_heat_1/attempt-old/result.json", {"score": 100})
    report = audit(c, f)
    assert report["protocol"]["matched"] is False
    assert report["protocol"]["mismatches"]
    assert report["common"]["contextgraph"]["mean_score"] == 20


def test_duplicate_tasks_rejected(tmp_path):
    scores = {"scienceworld_test_heat_1": 20}
    c = fixture(tmp_path / "c", "contextgraph", scores)
    f = fixture(tmp_path / "f", "foldagent", scores)
    write(c / "contextgraph-1/manifest.json", {"method": "contextgraph", "task_ids": list(scores)})
    with pytest.raises(ValueError, match="Overlapping"):
        audit(c, f)


def test_request_evidence_distinguishes_controller_and_token_zero(tmp_path):
    path = tmp_path / "requests.jsonl"
    rows = [{"input_ids": [0] * 100, "output_ids": [1] * 20, "finish_reason": "length"},
            {"input_ids": [2], "output_ids": [0] * 20,
             "structured_outputs": {"json": {"properties": {"candidate_indices": {}}}}}]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    stats = request_stats(path)
    assert stats["model_requests"] == 2 and stats["degenerate_requests"] == 1
    assert stats["graph_controller_requests"] == 1 and stats["output_tokens"] == 40
    assert request_stats(tmp_path / "missing")["available"] is False


def test_synced_vista_attempt_paths_are_resolved_locally(tmp_path):
    identity = "scienceworld_test_heat_1"
    c = fixture(tmp_path / "c", "contextgraph", {identity: 0})
    f = fixture(tmp_path / "f", "foldagent", {identity: 100})
    path = c / "contextgraph-0/instances" / identity / "result.json"
    row = json.loads(path.read_text())
    row["attempt"] = "/scratch/remote/attempt-current"
    write(path, row)
    tools = path.parent / "attempt-current/tools.jsonl"
    tools.write_text(json.dumps({"event": "step", "command": "look around", "info": {"score": 0}}))
    report = audit(c, f)
    assert report["cases"][identity]["contextgraph"]["steps_recorded"] == 1


@pytest.mark.parametrize("method,interval,expected_steps,expected_score", [
    ("contextgraph", 5, 84, 0), ("foldagent", 5, 90, 100), ("contextgraph", 0, 90, 100)])
def test_characterize_controller_calls_sharing_scienceworld_turn_cap(monkeypatch, method, interval, expected_steps, expected_score):
    """Controlled reproduction, not a claim about observed real task outcomes.

    Same action policy, no branches, success only at the 90th simulator action.
    Token budget is enlarged equally because the test tokenizer uses characters;
    this isolates the existing 100-model-turn vs 100-environment-step difference.
    """
    from scripts.eval_agent_benchmarks import config_for
    from tests.test_session_restart import Tokenizer
    from verl import DataProto
    class Simulator:
        def __init__(self, **kwargs): self.steps = 0
        def load(self, **kwargs): pass
        def reset(self): return "room", {"score": 0}
        def get_task_description(self): return "Complete ninety actions."
        def step(self, command):
            self.steps += 1
            score = 100 if self.steps == 90 else 0
            return f"Step {self.steps}", score, bool(score), {"score": score, "moves": self.steps}
        def close(self): pass
    class Client:
        calls = 0
        controllers = 0
        async def create_completion(self, ids, **kwargs):
            self.calls += 1
            schema = (kwargs.get("structured_outputs") or {}).get("json", {})
            if "candidate_indices" in schema.get("properties", {}):
                self.controllers += 1
                indices = schema["properties"]["candidate_indices"]["items"]["enum"]
                content = json.dumps({"action": "select", "candidate_indices": [indices[-1]], "summary": "", "relation": "causal"})
            else:
                content = "<function=action><parameter=command>look around</parameter></function>"
            tokens = list(map(ord, content)) + [0]
            return {"choices": [{"message": {"content": content, "raw_output_ids": tokens,
                                               "response_log_probs": [-0.125] * len(tokens)}}]}
    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=Simulator))
    config = config_for("scienceworld", method, 65536)
    config.actor_rollout_ref.rollout.response_length = 1_000_000
    config.actor_rollout_ref.rollout.plugin.val_response_length = 1_000_000
    config.actor_rollout_ref.rollout.plugin.consolidation_interval = interval
    task = DataProto()
    task.non_tensor_batch = {"ability": np.array(["ScienceWorld@real"], dtype=object),
                             "extra_info": np.array([{"task_name": "synthetic", "workflow": config.actor_rollout_ref.rollout.plugin.workflow}], dtype=object),
                             "uid": np.array(["synthetic"], dtype=object)}
    task.meta_info = {"generation_kwargs": {}}
    client = Client()
    context = SimpleNamespace(config=config, tokenizer=Tokenizer(), llm_client=client, is_train=False, global_step=0)
    module = importlib.import_module("agents.graph_agent_isolated" if method == "contextgraph" else "agents.fold_agent")
    output = asyncio.run(module.process_item(task, context))
    stats = output[0].extra_fields["env_stats"]
    assert stats["environment_steps"] == expected_steps
    assert stats["environment_score"] == expected_score
    controller_enabled = method == "contextgraph" and interval == 5
    assert client.controllers == (16 if controller_enabled else 0)
    assert client.calls == (100 if controller_enabled else 90)
