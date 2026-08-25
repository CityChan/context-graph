from agents.tool_protocol import canonicalize_tool_call_text


def test_canonicalizes_deepseek_parameter_separator_without_changing_semantics():
    raw = (
        "<function=action>\n"
        "<parameter>command>go to desk 1</parameter>\n"
        "</function>"
    )
    canonical, repairs = canonicalize_tool_call_text(raw)

    assert canonical == (
        "<function=action>\n"
        "<parameter=command>go to desk 1</parameter>\n"
        "</function>"
    )
    assert repairs == [{
        "kind": "parameter_name_separator",
        "parameter": "command",
    }]


def test_leaves_valid_tool_calls_and_plain_text_unchanged():
    valid = "<function=action><parameter=command>look</parameter></function>"
    assert canonicalize_tool_call_text(valid) == (valid, [])

    plain = "In prose, <parameter>command> is only an example."
    assert canonicalize_tool_call_text(plain) == (plain, [])
