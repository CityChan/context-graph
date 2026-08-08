import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.prompts import SEARCH_BRANCH_EXAMPLE, SEARCH_EXAMPLE, create_chat


def _load_actor_validate():
    source = Path("verl/workers/config/actor.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    actor_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActorConfig")
    validate = next(node for node in actor_class.body if isinstance(node, ast.FunctionDef) and node.name == "validate")
    module = ast.fix_missing_locations(ast.Module(body=[validate], type_ignores=[]))
    namespace = {}
    exec(compile(module, "verl/workers/config/actor.py", "exec"), namespace)
    return namespace["validate"]


def _actor_config(ppo_mini_batch_size):
    return SimpleNamespace(
        use_dynamic_bsz=False,
        rollout_n=2,
        ppo_mini_batch_size=ppo_mini_batch_size,
        ppo_micro_batch_size=None,
        ulysses_sequence_parallel_size=1,
    )


@pytest.mark.parametrize(
    ("workflow", "example"),
    [
        ("search", SEARCH_EXAMPLE),
        ("search_multi", SEARCH_EXAMPLE),
        ("search_branch", SEARCH_BRANCH_EXAMPLE),
        ("search_branch_multi", SEARCH_BRANCH_EXAMPLE),
    ],
)
def test_search_workflows_include_foldagent_few_shot_example(workflow, example):
    chat = create_chat("What is the answer?", workflow)
    content = chat[-1]["content"]

    assert content.startswith(example)
    assert content.index("What is the answer?") > len(example)


def test_actor_batch_validation_counts_rollouts():
    validate = _load_actor_validate()
    actor = _actor_config(ppo_mini_batch_size=8)

    validate(actor, n_gpus=1, train_batch_size=4)


def test_actor_batch_validation_reports_effective_batch_size():
    validate = _load_actor_validate()
    actor = _actor_config(ppo_mini_batch_size=9)

    with pytest.raises(ValueError, match=r"4 \* 2 = 8"):
        validate(actor, n_gpus=1, train_batch_size=4)
