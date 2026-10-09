import ast
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest


def _load_validation_metric_functions():
    path = Path("verl/trainer/ppo/metric_utils.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function_names = {"bootstrap_metric", "calc_maj_val", "process_validation_metrics"}
    functions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {
        "Any": Any,
        "Callable": Callable,
        "defaultdict": defaultdict,
        "np": np,
        "partial": partial,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["process_validation_metrics"]


process_validation_metrics = _load_validation_metric_functions()


def test_validation_grouping_normalizes_wrappers_without_splitting_prompts():
    metrics = process_validation_metrics(
        data_sources=["bcp", ["bcp"], np.array(["bcp"]), np.array("bcp")],
        sample_uids=["q1", ["q1"], np.array(["q1"]), np.array("q2")],
        infos_dict={"reward": [1.0, 0.0, 1.0, 0.0]},
    )
    assert set(metrics) == {"bcp"}
    assert metrics["bcp"]["reward"]["mean@3"] == pytest.approx(2 / 3)
    assert metrics["bcp"]["reward"]["mean@1"] == 0.0


def test_validation_grouping_preserves_distinct_composite_ids():
    metrics = process_validation_metrics(
        data_sources=[["bcp"]] * 3,
        sample_uids=[["q", 1], np.array(["q", 1], dtype=object), ["q", 2]],
        infos_dict={"reward": [1.0, 0.0, 1.0]},
    )
    assert metrics["bcp"]["reward"]["mean@2"] == 0.5
    assert metrics["bcp"]["reward"]["mean@1"] == 1.0


def test_process_validation_metrics_skips_missing_auxiliary_values():
    metrics = process_validation_metrics(
        data_sources=["browsecomp", "browsecomp", "browsecomp"],
        sample_uids=["question-1", "question-2", "question-3"],
        infos_dict={
            "reward": [1.0, 0.0, 1.0],
            "num_turns": [4, None, 8],
            "optional_metric": [None, None, None],
        },
    )

    assert metrics["browsecomp"]["reward"]["mean@1"] == pytest.approx(2 / 3)
    assert metrics["browsecomp"]["num_turns"]["mean@1"] == pytest.approx(6.0)
    assert "optional_metric" not in metrics["browsecomp"]


def test_process_validation_metrics_keeps_bootstrap_values_aligned_with_predictions():
    metrics = process_validation_metrics(
        data_sources=["browsecomp"] * 3,
        sample_uids=["question-1"] * 3,
        infos_dict={
            "reward": [1.0, None, 0.0],
            "pred": ["correct", "missing", "wrong"],
        },
    )

    reward_metrics = metrics["browsecomp"]["reward"]
    assert reward_metrics["mean@2"] == pytest.approx(0.5)
    assert reward_metrics["std@2"] == pytest.approx(0.5)
    assert "maj@2/mean" in reward_metrics
