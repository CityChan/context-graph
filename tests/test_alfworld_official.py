import asyncio
import json
import sys
import types

import numpy as np

from agents.tool_spec import alfworld_tool
from envs.alfworld_env import ALFWorldEnv
from scripts.make_alfworld_data import scan_alfworld_games


class _Item:
    def __init__(self, extra_info):
        self.non_tensor_batch = {
            "extra_info": np.array(extra_info, dtype=object),
        }


def test_init_uses_official_alfworld_wrappers(monkeypatch, tmp_path):
    captured = {}

    class EnvInfos:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AlfredDemangler:
        def __init__(self, shuffle=False):
            self.shuffle = shuffle

    class AlfredInfos:
        pass

    class FakeEnv:
        def reset(self):
            return (
                ["You are in a kitchen.\nYour task is to: put the apple on the table."],
                {"admissible_commands": [["look", "go to drawer 1"]]},
            )

        def close(self):
            pass

    gym_module = types.ModuleType("textworld.gym")

    def register_games(game_files, request_infos, **kwargs):
        captured["game_files"] = game_files
        captured["request_infos"] = request_infos
        captured["kwargs"] = kwargs
        return "fake-env"

    gym_module.register_games = register_games
    gym_module.make = lambda env_id: FakeEnv()

    textworld_module = types.ModuleType("textworld")
    textworld_module.__path__ = []
    textworld_module.EnvInfos = EnvInfos
    textworld_module.gym = gym_module

    official_module = types.ModuleType(
        "alfworld.agents.environment.alfred_tw_env"
    )
    official_module.AlfredDemangler = AlfredDemangler
    official_module.AlfredInfos = AlfredInfos

    monkeypatch.setitem(sys.modules, "textworld", textworld_module)
    monkeypatch.setitem(sys.modules, "textworld.gym", gym_module)
    monkeypatch.setitem(
        sys.modules,
        "alfworld.agents.environment.alfred_tw_env",
        official_module,
    )

    game_file = tmp_path / "game.tw-pddl"
    game_file.write_text("{}", encoding="utf-8")
    env = ALFWorldEnv(config=None, tokenizer=None, ability="ALFWorld@real")
    asyncio.run(env.init_env(_Item({"game_file": str(game_file)})))

    wrappers = captured["kwargs"]["wrappers"]
    assert isinstance(wrappers[0], AlfredDemangler)
    assert wrappers[0].shuffle is False
    assert wrappers[1] is AlfredInfos
    assert captured["request_infos"].kwargs["extras"] == ["gamefile"]
    assert env._admissible_commands == ["look", "go to drawer 1"]


def test_real_mode_executes_only_exact_official_command():
    class FakeEnv:
        def __init__(self):
            self.commands = []

        def step(self, commands):
            self.commands.append(commands)
            return (
                ["Moved."],
                [0],
                [False],
                {
                    "won": [False],
                    "admissible_commands": [["look", "go to drawer 1"]],
                },
            )

    env = ALFWorldEnv(config=None, tokenizer=None, ability="ALFWorld@real")
    env._tw_env = FakeEnv()
    env._admissible_commands = ["look", "go to drawer 1"]

    invalid = asyncio.run(
        env.run_action(
            "<function=action><parameter=command>go to drawer</parameter></function>"
        )
    )
    assert "Invalid action" in invalid["observation"]
    assert env._step_count == 0
    assert env._tw_env.commands == []

    asyncio.run(
        env.run_action(
            "<function=action><parameter=command>go to drawer 1</parameter></function>"
        )
    )
    assert env._step_count == 1
    assert env._tw_env.commands == [["go to drawer 1"]]


def test_non_environment_turns_do_not_consume_step_budget():
    env = ALFWorldEnv(config=None, tokenizer=None, ability="ALFWorld@real")
    env._admissible_commands = ["look"]

    asyncio.run(env.run_action("not a tool call"))
    asyncio.run(
        env.run_action(
            "<function=think><parameter=reasoning>plan</parameter></function>"
        )
    )
    asyncio.run(env.run_action("<function=action></function>"))

    assert env._step_count == 0
    assert env.stats["invalid_xml"] == 1
    assert env.stats["think"] == 1
    assert env.stats["empty_command"] == 1


def _write_game(root, task_type, trial, solvable=True):
    game_dir = root / task_type / trial
    game_dir.mkdir(parents=True)
    (game_dir / "game.tw-pddl").write_text(
        json.dumps({"solvable": solvable}),
        encoding="utf-8",
    )
    (game_dir / "traj_data.json").write_text(
        json.dumps({"task_type": task_type}),
        encoding="utf-8",
    )


def test_data_scan_matches_official_solvable_filters(tmp_path):
    data_root = tmp_path / "json_2.1.1"
    train = data_root / "train"
    _write_game(train, "pick_and_place_simple", "valid", solvable=True)
    _write_game(train, "pick_and_place_simple", "unsolvable", solvable=False)
    _write_game(
        train,
        "pick_and_place_with_movable_recep",
        "unsupported",
        solvable=True,
    )

    train_tasks, test_tasks = scan_alfworld_games(tmp_path)

    assert len(train_tasks) == 1
    assert train_tasks[0]["task_id"].endswith("_valid")
    assert test_tasks == []


def test_alfworld_prompt_can_expose_action_only_toolset():
    tools = alfworld_tool(action_only=True)
    assert [tool["function"]["name"] for tool in tools] == ["action"]
