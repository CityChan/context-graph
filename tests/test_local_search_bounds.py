from envs.local_search import keep_first_n_words


def test_document_bound_handles_long_whitespace_free_content():
    text = "x" * 10000
    bounded = keep_first_n_words(text, n=128, max_chars=500)

    assert bounded.startswith("x" * 500)
    assert len(bounded) < 600
    assert bounded.endswith("[Document is truncated.]")


def test_document_bound_limits_words_and_characters():
    text = " ".join(f"word{i}" for i in range(1000))
    bounded = keep_first_n_words(text, n=10, max_chars=1000)

    assert "word9" in bounded
    assert "word10" not in bounded
    assert bounded.endswith("[Document is truncated.]")
