import ast
from pathlib import Path

import numpy as np
import pytest


def _load_scalarize_reward_extra_info():
    path = Path("verl/trainer/ppo/ray_trainer.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected_names = {"_BATCH_LEVEL_REWARD_METRICS", "_scalarize_reward_extra_info"}
    selected_nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                target.id in selected_names
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in selected_names)
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["_scalarize_reward_extra_info"]


scalarize = _load_scalarize_reward_extra_info()


def test_scalarize_skips_missing_batch_level_values():
    assert scalarize("avg_score", [None, 0.25, 0.25]) == pytest.approx(0.25)
    assert scalarize("avg_score", [None, None]) is None


def test_scalarize_means_only_valid_per_sample_values():
    assert scalarize("reward_score", [1.0, None, 0.0]) == pytest.approx(0.5)


def test_scalarize_ignores_non_numeric_metadata():
    assert scalarize("pred", ["answer-a", "answer-b"]) is None
