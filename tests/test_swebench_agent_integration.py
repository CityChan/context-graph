"""Real agent loops with scripted completions and a fake container (no GPU/Docker)."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")
pytest.importorskip("tensordict")

from agents.fold_agent_code import process_item as fold_process
from agents.graph_agent_code_isolated import process_item as graph_process
from scripts.eval_swebench_verified import config_for, generate_one


class CharacterTokenizer:
    eos_token = "<end>"
    eos_token_id = 0

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False, **kwargs):
        text = "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


class Sandbox:
    instances = []

    def __init__(self, task, **kwargs):
        self.closed = False
        self.provenance = {"image_id": "fake"}
        self.calls = []
        self.instances.append(self)

    def start(self):
        pass

    def execute(self, code):
        self.calls.append(code)
        return 0, "Focused test passed"

    def patch(self):
        return "diff --git a/example.py b/example.py\n"

    def close(self):
        self.closed = True


@pytest.mark.parametrize("method", ["react", "foldagent", "contextgraph"])
def test_real_code_loop_exports_patch_and_cleans_container(method, tmp_path, monkeypatch):
    tokenizer = CharacterTokenizer()
    tool = "<function=python_exec><parameter=code>print('test')</parameter></function>"
    responses = [tool, "<function=finish><parameter=message>done</parameter></function>"]
    if method != "react":
        responses = ["<function=branch><parameter=description>inspect</parameter><parameter=prompt>inspect code</parameter></function>",
                     tool, "<function=return><parameter=message>inspected and tested</parameter></function>",
                     responses[-1]]

    class ScriptedClient:
        def __init__(self, *a, **kw):
            self.failed = False
            self.client = self

        async def aclose(self):
            pass

        async def create_completion(self, input_ids, **kwargs):
            text = responses.pop(0)
            ids = tokenizer.encode(text)
            return {"choices": [{"message": {"content": text, "raw_output_ids": ids,
                    "response_log_probs": [0.0] * len(ids)}}]}

    monkeypatch.setattr("envs.swebench_env.DockerSandbox", Sandbox)
    monkeypatch.setattr("scripts.eval_bcp_qwen38.TokenClient", ScriptedClient)
    task = {"instance_id": "django__django-1", "repo": "django/django", "base_commit": "a" * 40,
            "problem_statement": "Fix a bug mentioning GAIA and LocalSearch"}
    args = SimpleNamespace(method=method, max_turn=100, task_timeout=60, memory="8g", cpus=4,
                           seed=42, endpoint="http://unused", model="test")
    result, prediction = asyncio.run(generate_one(task, args, config_for(args), tokenizer, tmp_path,
                                                graph_process if method == "contextgraph" else fold_process))
    assert result["status"] == "generated", (result, list(tmp_path.rglob("error.txt")))
    assert prediction["model_patch"].startswith("diff")
    assert result["grading_status"] == "pending"
    assert "task_reward" not in result["env_stats"]
    assert Sandbox.instances[-1].closed
    assert len(Sandbox.instances[-1].calls) == 1
    trajectory = json.loads(next(tmp_path.rglob("trajectory.json")).read_text(encoding="utf-8"))
    assert trajectory["num_branches"] == (0 if method == "react" else 1)
    assert not responses
    tool_trace = [json.loads(line) for line in next(tmp_path.rglob("tool_trace.jsonl")).read_text().splitlines()]
    assert tool_trace == [{"call": 1, "code": "print('test')", "status": "completed",
                           "exit_code": 0, "output": "Focused test passed"}]


def test_agent_exception_cleans_owned_container_and_submits_no_partial_patch(tmp_path, monkeypatch):
    from envs.swebench_env import SWEVerifiedEnv

    class EmptyClient:
        def __init__(self, *a, **kw):
            self.failed = False
            self.client = self

        async def aclose(self):
            pass

    async def broken(item, context):
        env = SWEVerifiedEnv(context.config.actor_rollout_ref.rollout, None, item.non_tensor_batch["ability"][0])
        await env.init_env(item)
        raise RuntimeError("model request failed")

    monkeypatch.setattr("envs.swebench_env.DockerSandbox", Sandbox)
    monkeypatch.setattr("scripts.eval_bcp_qwen38.TokenClient", EmptyClient)
    task = {"instance_id": "django__django-1", "repo": "django/django", "base_commit": "a" * 40,
            "problem_statement": "Fix a bug"}
    args = SimpleNamespace(method="react", max_turn=100, task_timeout=60, memory="8g", cpus=4,
                           seed=42, endpoint="http://unused", model="test")
    result, prediction = asyncio.run(generate_one(task, args, config_for(args), CharacterTokenizer(), tmp_path, broken))
    assert result["status"] == "error" and prediction["model_patch"] == ""
    assert Sandbox.instances[-1].closed


def test_failed_tool_is_saved_without_trajectory_and_container_is_cleaned(tmp_path, monkeypatch):
    from envs.swebench_env import SWEVerifiedEnv

    class BrokenSandbox(Sandbox):
        def execute(self, code):
            raise RuntimeError("container transport failed")

    class Client:
        def __init__(self, *a, **kw):
            self.client = self
        async def aclose(self):
            pass

    async def broken(item, context):
        env = SWEVerifiedEnv(context.config.actor_rollout_ref.rollout, None,
                             item.non_tensor_batch["ability"][0])
        await env.init_env(item)
        await env.run_action("<function=python_exec><parameter=code>print(1)</parameter></function>")

    monkeypatch.setattr("envs.swebench_env.DockerSandbox", BrokenSandbox)
    monkeypatch.setattr("scripts.eval_bcp_qwen38.TokenClient", Client)
    task = {"instance_id": "django__django-1", "repo": "django/django", "base_commit": "a" * 40,
            "problem_statement": "Fix a bug"}
    args = SimpleNamespace(method="react", max_turn=100, task_timeout=60, memory="8g", cpus=4,
                           seed=42, endpoint="http://unused", model="test")
    result, prediction = asyncio.run(generate_one(task, args, config_for(args), CharacterTokenizer(), tmp_path, broken))
    assert result["status"] == "error" and prediction["model_patch"] == ""
    assert Sandbox.instances[-1].closed
    assert not list(tmp_path.rglob("trajectory.json"))
    trace = json.loads(next(tmp_path.rglob("tool_trace.jsonl")).read_text())
    assert trace["status"] == "error" and trace["code"] == "print(1)"
    assert "container transport failed" in trace["error"]
