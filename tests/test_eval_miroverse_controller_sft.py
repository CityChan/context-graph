import json

from scripts.eval_miroverse_controller_sft import (
    replay_response,
    score_response,
    validate_flat_decision,
)


def decision(action, indices, summary="", relation="semantic"):
    return json.dumps({
        "action": action,
        "candidate_indices": indices,
        "summary": summary,
        "relation": relation,
    })


def test_scores_replay_valid_structural_decision():
    gold = decision("add_edge", [0, 1], relation="causal")
    score = score_response(
        gold,
        gold,
        candidate_count=2,
        allow_pass=False,
        action_policy="structural",
    )
    assert score["parse_valid"]
    assert score["schema_valid"]
    assert score["replay_valid"]
    assert score["action_exact"]
    assert score["indices_exact"]
    assert score["structural_choice_exact"]
    assert score["decision_exact"]


def test_flat_schema_and_replay_contract_are_reported_separately():
    response = decision("merge", [0], summary="one item is insufficient")
    parsed = json.loads(response)
    validate_flat_decision(
        parsed,
        candidate_count=2,
        allow_pass=False,
        action_policy="structural",
    )
    valid, error = replay_response(
        response,
        candidate_count=2,
        allow_pass=False,
        action_policy="structural",
    )
    assert not valid
    assert error == "merge requires 2 to 6 candidate indices"


def test_pass_with_indices_exposes_known_conditional_error():
    response = decision("pass", [0])
    score = score_response(
        decision("pass", []),
        response,
        candidate_count=2,
        allow_pass=True,
        action_policy="balanced",
    )
    assert score["schema_valid"]
    assert not score["replay_valid"]
    assert score["error"] == "pass requires no candidate indices"


def test_structural_policy_rejects_select_with_multiple_candidates():
    response = decision("select", [0])
    score = score_response(
        decision("add_edge", [0, 1]),
        response,
        candidate_count=2,
        allow_pass=False,
        action_policy="structural",
    )
    assert score["parse_valid"]
    assert not score["schema_valid"]
    assert not score["replay_valid"]
    assert "unavailable" in score["error"]
