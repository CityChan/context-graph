from scripts.audit_graph_memory_smoke import task_diagnostics


def test_legacy_trajectory_exposes_stop_budget_and_bounded_errors():
    records = [dict(phase="final", format_error="missing function", response="x" * 3000,
                    output_ids=[1] * 5, max_tokens=5) for _ in range(5)]
    trajectory = dict(termination_reason="invalid_tool_limit", model_contexts=records,
                      env_stats=dict(main_len=24000, main_turn=15, invalid_memory=2),
                      memory_audit=[dict(phase="memory_memorize", request_index=3, error="invalid JSON")])
    result = task_diagnostics(28, trajectory)
    assert result["stop"] == "invalid_tool_limit" and result["source_index"] == 28
    assert result["format_error_events"] == 5
    assert len(result["last_format_errors"]) == 3
    assert result["last_format_errors"][0]["at_output_limit"]
    assert len(result["last_format_errors"][0]["response_tail"]) == 1600
    assert result["memory_error_events"] == 1
    assert result["last_memory_errors"][0]["request_index"] == 3
    assert result["last_memory_errors"][0]["response_tail"] == "x" * 1600
    assert result["stats"]["invalid_memory"] == 2


def test_token_limit_without_format_errors_stays_visible():
    result = task_diagnostics(70, dict(termination_reason="token_limit",
                                     env_stats=dict(main_len=24576, hit_token_limit=1)))
    assert result["stop"] == "token_limit"
    assert result["stats"]["main_len"] == 24576
    assert result["format_error_events"] == 0 and result["last_memory_errors"] == []
