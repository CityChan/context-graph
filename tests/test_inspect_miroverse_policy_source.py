from scripts.inspect_miroverse_policy_source import assistant_signature, inspect_records, mcp_calls


def test_assistant_signature_distinguishes_tool_protocols():
    assert assistant_signature("<function=search></function>") == "function:search"
    assert assistant_signature("think <search><query>x</query></search>") == "tags:search+query"
    assert assistant_signature("final answer") == "plain"


def test_extracts_mcp_server_tool_and_arguments():
    content = (
        "reason <use_mcp_tool><server_name>browsing-agent</server_name>"
        "<tool_name>search_and_browse</tool_name><arguments>{\"subtask\":\"x\"}</arguments>"
        "</use_mcp_tool>"
    )
    assert mcp_calls(content) == [
        ("browsing-agent", "search_and_browse", '{"subtask":"x"}')
    ]


def test_inspect_records_reports_policy_shapes():
    records = [{
        "query": "Q",
        "answer": "A",
        "messages": [
            {"role": "system", "content": "tools"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "<search><query>x</query></search>"},
            {"role": "user", "content": "result"},
            {"role": "assistant", "content": "A"},
        ],
    }]
    summary = inspect_records(records, max_samples=0, examples_per_signature=1)
    assert summary["rows"] == 1
    assert summary["assistant_signatures"] == {"tags:search+query": 1, "plain": 1}
    assert summary["final_assistant_signatures"] == {"plain": 1}
    assert summary["role_transitions"]["assistant->user"] == 1
