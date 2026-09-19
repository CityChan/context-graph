import json
from pathlib import Path

from scripts.eval_miroverse_controller_sft import (
    replay_response,
    score_response,
    validate_flat_decision,
)


def test_disables_torch_compile_before_importing_vllm():
    source = Path("scripts/eval_miroverse_controller_sft.py").read_text(encoding="utf-8")
    disable = 'os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")'
    assert disable in source
    assert source.index(disable) < source.index("from vllm import LLM, SamplingParams")


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


def test_schema_failure_does_not_hide_runtime_replay_or_action_agreement():
    gold = decision("prune", [0])
    predicted = json.dumps({
        "action": "prune",
        "candidate_indices": [0],
        "summary": "",
        "relation": "semantic",
        "explanation": "extra field rejected by the strict schema",
    })
    score = score_response(
        gold,
        predicted,
        candidate_count=2,
        allow_pass=False,
        action_policy="structural",
    )
    assert score["parse_valid"]
    assert not score["schema_valid"]
    assert score["replay_valid"]
    assert score["action_exact"]
    assert score["indices_exact"]
    assert score["schema_error"] == "response fields do not exactly match the controller schema"
    assert score["replay_error"] is None


def test_missing_candidate_indices_reports_schema_and_replay_errors():
    predicted = json.dumps({
        "action": "prune",
        "summary": "",
        "relation": "semantic",
    })
    score = score_response(
        decision("prune", [0]),
        predicted,
        candidate_count=2,
        allow_pass=False,
        action_policy="structural",
    )
    assert score["parse_valid"]
    assert not score["schema_valid"]
    assert not score["replay_valid"]
    assert score["action_exact"]
    assert score["schema_error"] == "response fields do not exactly match the controller schema"
    assert score["replay_error"] == "candidate_indices is not an array"
