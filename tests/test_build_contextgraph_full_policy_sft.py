from scripts.build_contextgraph_full_policy_sft import branch_trajectory_to_row, function_counts


def branch_messages():
    return [
        {"role": "system", "content": "research"},
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": "<function=branch>\n</function>"},
        {"role": "user", "content": "ROLE CHANGE: `MODE: BRANCH`\nFind evidence."},
        {"role": "assistant", "content": "<function=search>\n<parameter=query>x</parameter>\n</function>"},
        {"role": "user", "content": "search results"},
        {"role": "assistant", "content": "<function=open_page>\n<parameter=docid>d1</parameter>\n</function>"},
        {"role": "user", "content": "page evidence"},
        {"role": "assistant", "content": "<function=return>\n<parameter=message>found it</parameter>\n</function>"},
    ]


def test_accepts_complete_branch_policy_trajectory():
    row = branch_trajectory_to_row(
        {"agent_name": "#0-evidence", "messages": branch_messages()},
        task_id="task",
        query_hash="hash",
        teacher_model="deepseek",
        require_open_page=True,
    )
    assert row is not None
    assert row["policy_role"] == "branch"
    assert (row["search_calls"], row["open_page_calls"], row["return_calls"]) == (1, 1, 1)


def test_rejects_branch_without_open_page_or_terminal_return():
    no_open = branch_messages()
    del no_open[6:8]
    reasons = []
    assert branch_trajectory_to_row(
        {"messages": no_open},
        task_id="task",
        query_hash="hash",
        teacher_model="deepseek",
        require_open_page=True,
        rejection_reasons=reasons,
    ) is None
    assert reasons == ["branch_no_open_page"]
    no_return = branch_messages()[:-1]
    reasons = []
    assert branch_trajectory_to_row(
        {"messages": no_return},
        task_id="task",
        query_hash="hash",
        teacher_model="deepseek",
        require_open_page=True,
        rejection_reasons=reasons,
    ) is None
    assert reasons == ["branch_no_terminal_return"]


def test_function_counts_reports_all_policy_tools():
    messages = [
        {"role": "assistant", "content": "<function=branch></function><function=finish></function>"},
        {"role": "assistant", "content": "<function=search></function><function=open_page></function><function=return></function>"},
    ]
    counts = function_counts(messages)
    assert {name: counts[name] for name in ("branch", "finish", "search", "open_page", "return")} == {
        "branch": 1,
        "finish": 1,
        "search": 1,
        "open_page": 1,
        "return": 1,
    }
