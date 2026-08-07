#!/usr/bin/env python3
"""Single-node smoke test for the official ALFWorld TextWorld harness.

This intentionally avoids model loading. It verifies that a solvable game can
be initialized with ALFWorld's official demangler, that the displayed
admissible commands contain executable natural-language entity names, and
that an exact command advances TextWorld without parser or environment errors.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.alfworld_env import ALFWorldEnv
from scripts.make_alfworld_data import scan_alfworld_games


def _item_for_game(game: dict) -> SimpleNamespace:
    extra_info = {
        "game_file": game["game_file"],
        "task_desc": game["task_desc"],
        "problem_statement": game["task_desc"],
        "task_type": game["task_type"],
        "workflow": "alfworld",
    }
    return SimpleNamespace(
        non_tensor_batch={
            "extra_info": np.array(extra_info, dtype=object),
        }
    )


def _choose_game(data_root: str, game_file: str | None) -> dict:
    if game_file:
        return {
            "game_file": os.path.abspath(game_file),
            "task_desc": "ALFWorld official harness smoke",
            "task_type": "explicit_game",
        }

    train, test = scan_alfworld_games(
        data_root,
        max_train=1,
        max_test=1,
        seed=42,
    )
    candidates = train or test
    if not candidates:
        raise RuntimeError(f"No officially solvable ALFWorld games found under {data_root}")
    return candidates[0]


def _xml_action(command: str) -> str:
    return (
        "<function=action>"
        f"<parameter=command>{command}</parameter>"
        "</function>"
    )


async def run_smoke(args: argparse.Namespace) -> None:
    game = _choose_game(args.alfworld_data, args.game_file)
    print(f"game_file={game['game_file']}")
    print(f"task_type={game['task_type']}")

    env = ALFWorldEnv(config=None, tokenizer=None, ability="ALFWorld@real")
    await env.init_env(_item_for_game(game))
    if env.env_fail:
        raise RuntimeError("ALFWorld environment initialization failed")

    commands = list(env._admissible_commands)
    print(f"task={env.instance_info.get('problem_statement', '')}")
    print(f"initial_admissible_count={len(commands)}")
    print(f"initial_admissible_sample={commands[:10]}")
    if not commands:
        raise AssertionError("Official environment returned no admissible commands")
    raw_ids = [command for command in commands if "_bar_" in command]
    if raw_ids:
        raise AssertionError(f"AlfredDemangler did not run; raw entity IDs remain: {raw_ids[:3]}")

    command = "look" if "look" in commands else commands[0]
    before = env._step_count
    result = await env.run_action(_xml_action(command))
    after = env._step_count
    print(f"executed={command!r}")
    print(f"step_count={before}->{after}")
    print(f"observation={result.get('observation', '')[:800]}")

    if after != before + 1:
        raise AssertionError("An exact admissible command did not consume exactly one TextWorld step")
    if env.stats["admissible_exact"] != 1:
        raise AssertionError(f"Expected one exact admissible action, stats={dict(env.stats)}")
    if env.stats["admissible_miss"] or env.stats["env_error"]:
        raise AssertionError(f"Official command execution failed, stats={dict(env.stats)}")

    post_commands = list(env._admissible_commands)
    post_raw_ids = [command for command in post_commands if "_bar_" in command]
    if post_raw_ids:
        raise AssertionError(f"Raw entity IDs appeared after stepping: {post_raw_ids[:3]}")
    print(f"post_admissible_count={len(post_commands)}")
    print(f"stats={dict(env.stats)}")
    env.close()
    print("SMOKE PASS: official ALFWorld demangling and exact-action execution work")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alfworld-data",
        default=os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")),
    )
    parser.add_argument("--game-file", default=None)
    args = parser.parse_args()
    asyncio.run(run_smoke(args))


if __name__ == "__main__":
    main()
