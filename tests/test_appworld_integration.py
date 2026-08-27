import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from envs.appworld_env import AppWorldEnv
from scripts.make_appworld_data import collect_train_tasks, to_row


class _Evaluation:
    success = True


class _Task:
    instruction = "Create a reminder and notify the supervisor."
    app_descriptions = {"todoist": "Task management"}


class _FakeWorld:
    def __init__(self, task_id, experiment_name):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.task = _Task()
        self.codes = []
        self.closed = False

    def execute(self, code):
        self.codes.append(code)
        return "created"

    def task_completed(self):
        return "complete_task" in self.codes[-1]

    def evaluate(self):
        return _Evaluation()

    def close(self):
        self.closed = True


class _Item:
    non_tensor_batch = {
        "extra_info": np.array([{"task_id": "train_001", "split": "train"}], dtype=object)
    }


def _config():
    return SimpleNamespace(plugin=SimpleNamespace())


def test_appworld_executes_multiline_code_and_uses_state_evaluator(monkeypatch):
    monkeypatch.setitem(sys.modules, "appworld", types.SimpleNamespace(AppWorld=_FakeWorld))
    env = AppWorldEnv(_config(), tokenizer=None, ability="AppWorld@train")
    asyncio.run(env.init_env(_Item()))
    assert "Create a reminder" in env.instance_info["problem_statement"]
    result = asyncio.run(env.run_action(
        "<function=action><parameter=code>x = 1\nprint(x)\napis.supervisor.complete_task()</parameter></function>"
    ))
    assert "completion was submitted" in result["observation"]
    assert env._world.codes[-1].startswith("x = 1\n")
    assert env.is_finish
    assert asyncio.run(env.get_reward(_Item(), [], None))[1] == 1.0


def test_appworld_rejects_non_train_rows():
    class _DevItem:
        non_tensor_batch = {
            "extra_info": np.array([{"task_id": "dev_001", "split": "dev"}], dtype=object)
        }

    env = AppWorldEnv(_config(), tokenizer=None, ability="AppWorld@train")
    asyncio.run(env.init_env(_DevItem()))
    assert env.env_fail
    assert env.stats["rejected_non_train_task"] == 1


def test_appworld_data_is_deterministic_and_train_only(monkeypatch):
    fake = types.SimpleNamespace(load_task_ids=lambda split: ["b", "a", "c"] if split == "train" else [])
    monkeypatch.setitem(sys.modules, "appworld", fake)
    tasks = collect_train_tasks(seed=7)
    assert tasks == collect_train_tasks(seed=7)
    assert {task["split"] for task in tasks} == {"train"}
    row = to_row(tasks[0], "appworld_graph")
    assert row["ability"] == "AppWorld@train"
    assert row["extra_info"]["workflow"] == "appworld_graph"


def test_appworld_dispatch_graph_prompt_and_scripts():
    pytest.importorskip("omegaconf")
    from agents.prompts import create_chat
    from agents.utils import select_env
    from scripts.eval_interactive import WORKFLOWS

    assert select_env("AppWorld@train", _config()) is AppWorldEnv
    assert WORKFLOWS["appworld_graph"] == "graph"
    chat = create_chat("Create a reminder.", "appworld_graph")
    text = "\n".join(message["content"] for message in chat)
    assert "AppWorld" in text
    assert "code (string, required)" in text
    assert "GRAPH ACTION MODE" in text

    generator = Path("scripts/generate_ctxgraph_sft_deepseek_v4_interactive_8node.sh").read_text()
    submitter = Path("scripts/submit_full_ctxgraph_sft_appworld_deepseek_v4.sh").read_text()
    assert "DOMAIN=appworld" in generator
    assert "appworld_graph_train.parquet" in generator
    assert "APPWORLD_NUM_WORKERS=${APPWORLD_NUM_WORKERS:-1}" in generator
    assert "APPWORLD_ROOT=${APPWORLD_ROOT:-$SCRATCH/" in generator
    assert "--count-only" in submitter
    assert 'if [ "$DRY_RUN" = "1" ]' in submitter
