import json
import sys

import pytest

from scripts.audit_counterfactual_graph_credit import audit_results, main


def test_audit_counterfactual_graph_credit_deduplicates_episode_streams(tmp_path):
    event = {
        "seq": 4,
        "graph_rpo_credit_backend": "old_policy_counterfactual_qa",
        "graph_rpo_counterfactual_before_responses": ["<answer>wrong</answer>"],
        "graph_rpo_counterfactual_after_responses": ["<answer>right</answer>"],
        "graph_rpo_counterfactual_before_rewards": [0.0],
        "graph_rpo_counterfactual_after_rewards": [1.0],
        "graph_rpo_delta": 0.25,
    }
    record = {"gen_uid": "same-episode", "graph_trace": {"events": [event]}}
    path = tmp_path / "rollouts.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    report = audit_results([path])

    assert report["summary"]["unique_episodes"] == 1
    assert report["summary"]["counterfactual_edits"] == 1
    assert report["summary"]["probe_responses"] == 2
    assert report["summary"]["answer_tag_rate"] == 1.0
    assert report["summary"]["positive_probe_rewards"] == 1
    assert report["summary"]["paired_sample_reward_differences"] == 1
    assert report["summary"]["nonzero_edit_deltas"] == 1
    assert report["summary"]["delta_abs_sum"] == 0.25


def test_audit_counterfactual_graph_credit_reports_truncated_probe(tmp_path):
    event = {
        "seq": 5,
        "graph_rpo_credit_backend": "old_policy_counterfactual_qa",
        "graph_rpo_counterfactual_before_responses": ["Wait, maybe"],
        "graph_rpo_counterfactual_after_responses": ["<answer>still wrong</answer>"],
        "graph_rpo_counterfactual_before_rewards": [0.0],
        "graph_rpo_counterfactual_after_rewards": [0.0],
        "graph_rpo_delta": 0.0,
    }
    path = tmp_path / "rollouts.jsonl"
    path.write_text(
        json.dumps({"gen_uid": "episode", "graph_trace": {"events": [event]}}),
        encoding="utf-8",
    )

    report = audit_results([path])

    assert report["summary"]["answer_tag_rate"] == 0.5
    assert report["summary"]["nonzero_edit_deltas"] == 0
    assert report["malformed_response_samples"][0]["response_tail"] == "Wait, maybe"


def test_audit_graph_credit_reports_semantic_noops_clipping_and_ops(tmp_path):
    before = {
        "nodes": [{"id": "n0", "status": "active"}],
        "edges": [],
        "root_id": "n0",
        "active_node_id": "n0",
        "counters": {"operation_count": 1},
    }
    event = {
        "seq": 6,
        "op": "select",
        "graph_rpo_credit_backend": "reference_answer_likelihood",
        "before_state": before,
        "after_state": {**before, "counters": {"operation_count": 2}},
        "graph_rpo_delta_unclipped": 0.5,
        "graph_rpo_delta": 0.25,
        "graph_rpo_outcome_gated": False,
    }
    gated_event = {
        **event,
        "seq": 7,
        "op": "merge",
        "graph_rpo_delta_unclipped": None,
        "graph_rpo_delta": 0.0,
        "graph_rpo_outcome_gated": True,
    }
    path = tmp_path / "reference.jsonl"
    path.write_text(
        json.dumps(
            {
                "gen_uid": "episode",
                "graph_trace": {"events": [event, gated_event]},
            }
        ),
        encoding="utf-8",
    )

    report = audit_results(
        [path], backend="reference_answer_likelihood"
    )
    summary = report["summary"]

    assert summary["selected_credit_edits"] == 2
    assert summary["scored_credit_edits"] == 1
    assert summary["outcome_gated_edits"] == 1
    assert summary["semantic_noop_edits"] == 2
    assert summary["semantic_state_change_edits"] == 0
    assert summary["positive_edit_deltas"] == 1
    assert summary["zero_edit_deltas"] == 1
    assert summary["zero_scored_edit_deltas"] == 0
    assert summary["clipped_edit_deltas"] == 1
    assert summary["clipped_edit_deltas_by_op"] == {"select": 1}
    assert summary["clip_rate"] == 1.0
    assert summary["clip_rate_all"] == 0.5
    assert summary["nonzero_edit_rate"] == 1.0
    assert summary["nonzero_edit_rate_all"] == 0.5
    assert summary["edits_by_op"] == {"merge": 1, "select": 1}
    assert summary["scored_edits_by_op"] == {"select": 1}
    assert summary["outcome_gated_edits_by_op"] == {"merge": 1}
    assert summary["delta_sum_by_op"] == {"merge": 0.0, "select": 0.25}
    assert summary["raw_delta_distribution"] == {
        "count": 1,
        "min": 0.5,
        "max": 0.5,
        "mean": 0.5,
        "p50": 0.5,
        "p75": 0.5,
        "p80": 0.5,
        "p90": 0.5,
        "p95": 0.5,
        "p99": 0.5,
    }
    assert summary["raw_abs_delta_distribution"]["p80"] == 0.5
    assert summary["raw_abs_delta_distribution_by_op"]["select"]["p80"] == 0.5
    assert report["semantic_noop_samples"][0]["seq"] == 6


def test_audit_uses_scaled_preclip_delta_for_clip_detection(tmp_path):
    event = {
        "seq": 8,
        "op": "merge",
        "graph_rpo_credit_backend": "reference_answer_likelihood",
        "before_hash": "before",
        "after_hash": "after",
        "graph_rpo_delta_unclipped": 0.5,
        "graph_rpo_delta_scale": 5.0,
        "graph_rpo_delta_scaled_unclipped": 0.1,
        "graph_rpo_delta": 0.1,
        "graph_rpo_outcome_gated": False,
    }
    path = tmp_path / "scaled.jsonl"
    path.write_text(
        json.dumps({"gen_uid": "episode", "graph_trace": {"events": [event]}}),
        encoding="utf-8",
    )

    summary = audit_results(
        [path], backend="reference_answer_likelihood"
    )["summary"]

    assert summary["clipped_edit_deltas"] == 0
    assert summary["clip_rate"] == 0.0
    assert summary["delta_scale_distribution"] == {
        "count": 1,
        "min": 5.0,
        "max": 5.0,
        "mean": 5.0,
        "p50": 5.0,
        "p75": 5.0,
        "p80": 5.0,
        "p90": 5.0,
        "p95": 5.0,
        "p99": 5.0,
    }
    assert summary["raw_abs_delta_distribution"]["p80"] == 0.5


def test_audit_cli_enforces_expected_delta_scale(tmp_path, monkeypatch):
    event = {
        "seq": 9,
        "op": "merge",
        "graph_rpo_credit_backend": "reference_answer_likelihood",
        "before_hash": "before",
        "after_hash": "after",
        "graph_rpo_delta_unclipped": 0.8,
        "graph_rpo_delta_scale": 4.0,
        "graph_rpo_delta_scaled_unclipped": 0.2,
        "graph_rpo_delta": 0.2,
        "graph_rpo_outcome_gated": False,
    }
    path = tmp_path / "scaled.jsonl"
    path.write_text(
        json.dumps({"gen_uid": "episode", "graph_trace": {"events": [event]}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_counterfactual_graph_credit.py",
            str(path),
            "--backend",
            "reference_answer_likelihood",
            "--expected-delta-scale",
            "4.0",
        ],
    )
    main()

    monkeypatch.setattr(sys, "argv", [*sys.argv[:-1], "5.0"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
