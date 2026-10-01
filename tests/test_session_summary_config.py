import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.rollout_status import validate_session_summary


def test_restart_is_opt_in_and_rejects_invalid_limits():
    validate_session_summary(SimpleNamespace())
    validate_session_summary(SimpleNamespace(enable_summary=False))
    validate_session_summary(SimpleNamespace(enable_summary=True))
    with pytest.raises(ValueError, match="summary_max_tokens"):
        validate_session_summary(SimpleNamespace(enable_summary=True, summary_max_tokens=0))


@pytest.mark.parametrize("name", ["fold_agent", "fold_agent_code", "graph_agent", "graph_agent_isolated", "graph_agent_code_isolated"])
def test_executors_validate_summary_before_opening_environment(name):
    path = Path(__file__).resolve().parents[1] / "agents" / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    check = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "validate_session_summary"]
    init = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "env.init_env"]
    assert len(check) == 1
    assert check[0].lineno < init[0].lineno
