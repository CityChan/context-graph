import asyncio

from envs.local_search import AsyncSearchClient, keep_first_n_words


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


def test_search_client_accepts_round_robin_server_pool():
    client = AsyncSearchClient("http://search-0:18999,http://search-1:18999")

    assert client.base_urls == ["http://search-0:18999", "http://search-1:18999"]
    assert client._round_robin_client() is client._clients[0]
    assert client._round_robin_client() is client._clients[1]
    assert client._round_robin_client() is client._clients[0]

    asyncio.run(client.close())
