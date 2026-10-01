from agents.prompts import (
    SEARCH_SYSTEM_PROMPT_GRAPH_CONTROLLER,
    SEARCH_USER_PROMPT_GRAPH_CONTROLLER,
)
from agents.prompts_code import _CODE_GRAPH_CONTROLLER_ADDENDUM


def test_controller_prompt_restores_deep_research_workflow():
    system_prompt = SEARCH_SYSTEM_PROMPT_GRAPH_CONTROLLER
    user_prompt = SEARCH_USER_PROMPT_GRAPH_CONTROLLER

    for instruction in (
        "Construct & Plan",
        "Branch & Investigate",
        "Verify & Iterate",
        "Verification Checklist",
        "5-15 tool calls",
    ):
        assert instruction in system_prompt

    assert "Construct your research plan" in user_prompt
    assert "Branch sub-tasks to explore independent angles" in user_prompt
    assert "unlimited thinking budget" not in user_prompt
    assert "if the remaining budget permits" in user_prompt
    assert "Explicit harness instructions to finalize take precedence" in user_prompt


def test_controller_prompt_keeps_graph_protocol_isolated():
    combined = (
        SEARCH_SYSTEM_PROMPT_GRAPH_CONTROLLER
        + SEARCH_USER_PROMPT_GRAPH_CONTROLLER
    )

    assert "[GRAPH ACTION MODE]" in combined
    assert "controller-requested JSON actions" in combined
    assert "Never emit graph-management XML actions" in combined
    assert "<function=merge>" not in combined
    assert "<function=prune>" not in combined


def test_code_controller_prompt_keeps_deep_scientific_workflow():
    assert "scientific workflow" in _CODE_GRAPH_CONTROLLER_ADDENDUM
    assert "Consolidate complementary analyses" in _CODE_GRAPH_CONTROLLER_ADDENDUM
    assert "select` only for a genuine change" in _CODE_GRAPH_CONTROLLER_ADDENDUM
    assert "inspect-plan-execute-verify" in _CODE_GRAPH_CONTROLLER_ADDENDUM
