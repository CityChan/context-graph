import json

from scripts.prepare_miroverse_retrieval_corpus import build_corpus


def call(server, tool, arguments, observation):
    return {
        "messages": [
            {"role": "system", "content": "tools"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "content": (
                    "research <use_mcp_tool>"
                    f"<server_name>{server}</server_name>"
                    f"<tool_name>{tool}</tool_name>"
                    f"<arguments>{json.dumps(arguments)}</arguments>"
                    "</use_mcp_tool>"
                ),
            },
            {"role": "user", "content": observation},
            {"role": "assistant", "content": "final"},
        ]
    }


def test_extracts_supported_search_observation_with_provenance():
    rows, audit = build_corpus(
        [
            call(
                "tool-google-search",
                "scrape",
                {"url": "https://example.com/page"},
                "Example evidence text with enough characters to be retained in corpus.",
            )
        ],
        max_samples=0,
        min_characters=20,
        max_characters=1000,
    )
    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.com/page"
    assert rows[0]["source"].endswith("MiroVerse-MuSiQue")
    assert rows[0]["tool_name"] == "scrape"
    assert audit["counters"]["accepted_documents"] == 1


def test_deduplicates_normalized_observations_and_counts_duplicates():
    first = call(
        "browsing-agent",
        "search_and_browse",
        {"subtask": "find evidence"},
        "The same sufficiently long evidence appears in this observation.",
    )
    second = call(
        "agent-browsing",
        "search_and_browse",
        {"subtask": "repeat evidence"},
        "The same   sufficiently long evidence\nappears in this observation.",
    )
    rows, audit = build_corpus(
        [first, second],
        max_samples=0,
        min_characters=20,
        max_characters=1000,
    )
    assert len(rows) == 1
    assert rows[0]["duplicate_count"] == 1
    assert audit["counters"]["duplicate_observations"] == 1


def test_rejects_non_search_tool_observations():
    rows, audit = build_corpus(
        [call("tool-code", "run_python_code", {"code_block": "print(1)"}, "long code output")],
        max_samples=0,
        min_characters=5,
        max_characters=1000,
    )
    assert rows == []
    assert audit["unsupported_tool_counts"] == {"tool-code/run_python_code": 1}


def test_synthetic_urls_are_stable_when_observation_has_no_url():
    rows, audit = build_corpus(
        [
            call(
                "tool-google-search",
                "google_search",
                {"q": "example"},
                "A sufficiently detailed search result without a source URL.",
            )
        ],
        max_samples=0,
        min_characters=20,
        max_characters=1000,
    )
    assert rows[0]["url"].startswith("miroverse://observation/miroverse_obs_")
    assert audit["counters"]["synthetic_urls"] == 1
