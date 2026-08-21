import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from envs.scienceworld_env import ScienceWorldEnv


class _FakeSimulator:
    def __init__(self, taskName="", envStepLimit=100):
        self.task_name = taskName
        self.limit = envStepLimit
        self.closed = False

    def load(self, taskName, variationIdx, simplificationStr, generateGoldPath):
        self.task_name = taskName
        self.variation_idx = variationIdx

    def reset(self):
        return "You are in the workshop with a battery and wire.", {"valid": True}

    def get_task_description(self):
        return "Build a working circuit."

    def step(self, command):
        if command == "connect battery to wire":
            return "The circuit is complete.", 100, True, {"valid": True, "score": 100}
        return "Nothing happens.", 0, False, {"valid": False, "score": 0}

    def close(self):
        self.closed = True


class _Item:
    non_tensor_batch = {
        "extra_info": np.array([{
            "task_id": "scienceworld_train_test_0",
            "task_name": "test-task",
            "variation_idx": 3,
            "simplification": "",
        }], dtype=object)
    }


def _config():
    return SimpleNamespace(plugin=SimpleNamespace(scienceworld_max_steps=12))


def test_scienceworld_env_executes_and_scores(monkeypatch):
    monkeypatch.setitem(sys.modules, "scienceworld", types.SimpleNamespace(ScienceWorldEnv=_FakeSimulator))
    env = ScienceWorldEnv(_config(), tokenizer=None, ability="ScienceWorld@real")
    asyncio.run(env.init_env(_Item()))

    assert not env.env_fail
    assert "Build a working circuit" in env.instance_info["problem_statement"]
    result = asyncio.run(env.run_action(
        "<function=action><parameter=command>connect battery to wire</parameter></function>"
    ))
    assert "completed successfully" in result["observation"]
    assert env.is_finish
    assert asyncio.run(env.get_reward(_Item(), [], None))[1] == 1.0


def test_scienceworld_dispatch_and_graph_prompt():
    pytest.importorskip("omegaconf")
    from agents.prompts import create_chat
    from agents.utils import select_env

    assert select_env("ScienceWorld@real", _config()) is ScienceWorldEnv
    chat = create_chat("Test conductivity.\n\nInitial observation:\nA room.", "scienceworld_graph")
    text = "\n".join(message["content"] for message in chat)
    assert "ScienceWorld" in text
    assert "<function=action>" not in text
    assert "BEGIN FUNCTION" in text
    assert "merge" in text


def test_qwen36_interactive_smoke_is_two_domain_train_only_array():
    path = Path("scripts/smoke_interactive_ctxgraph_qwen36_27b_1node.sh")
    text = path.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-1" in text
    assert "Qwen/Qwen3.6-27B" in text
    assert "DOMAIN=alfworld" in text
    assert "DOMAIN=scienceworld" in text
    assert "alfworld_graph_real_train.parquet" in text
    assert "scienceworld_graph_train.parquet" in text
    assert "--save-messages" in text
    assert "--min-structural-graph-ops 1" in text
    assert "--max-invalid-graph-ops 0" in text
    assert "scienceworld==$SCIENCEWORLD_VERSION" in text


def test_interactive_evaluator_supports_both_graph_workflows():
    pytest.importorskip("omegaconf")
    from scripts.eval_interactive import WORKFLOWS

    assert WORKFLOWS["alfworld_graph"] == "graph"
    assert WORKFLOWS["scienceworld_graph"] == "graph"
