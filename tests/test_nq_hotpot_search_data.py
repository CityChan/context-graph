import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

from envs.local_search import LocalSearch, searchr1_em_score
from envs.wiki18_search_server import BatchedSearchEngine, split_contents, truncate_words
from scripts.convert_faiss_index_to_numpy import convert
from scripts.prepare_nq_hotpot_search_data import balanced_sample, convert_frame


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "nq-1",
                "question": "Who wrote it?",
                "golden_answers": ["The Author", "Author"],
                "data_source": "nq",
                "reward_model": {"ground_truth": {"target": ["The Author"]}},
            },
            {
                "id": "hp-1",
                "question": "Where was it built?",
                "golden_answers": ["Paris, France"],
                "data_source": "hotpotqa",
                "reward_model": {"ground_truth": {"target": ["Paris"]}},
            },
            {
                "id": "pop-1",
                "question": "Ignored?",
                "golden_answers": ["Yes"],
                "data_source": "popqa",
                "reward_model": {"ground_truth": {"target": ["Yes"]}},
            },
        ]
    )


def test_conversion_filters_sources_and_preserves_aliases():
    converted, counts = convert_frame(_raw_frame(), "train", seed=7)
    assert set(converted.data_source) == {"searchR1_nq", "searchR1_hotpotqa"}
    assert counts["skipped:popqa"] == 1
    nq = converted.loc[converted.data_source == "searchR1_nq"].iloc[0]
    assert nq.ability == "LocalSearch"
    assert nq.extra_info["reward_mode"] == "searchr1_em"
    assert nq.extra_info["answer_aliases"] == ["The Author", "Author"]


def test_balanced_validation_sample_is_deterministic():
    converted, _ = convert_frame(pd.concat([_raw_frame()] * 4, ignore_index=True), "validation", seed=3)
    first = balanced_sample(converted, per_source=2, seed=9)
    second = balanced_sample(converted, per_source=2, seed=9)
    assert first.equals(second)
    assert first.data_source.value_counts().to_dict() == {
        "searchR1_nq": 2,
        "searchR1_hotpotqa": 2,
    }


def test_searchr1_em_accepts_any_alias_with_standard_normalization():
    assert searchr1_em_score(["The Author", "A. Writer"], "author")
    assert searchr1_em_score(["Paris, France"], "Paris France")
    assert not searchr1_em_score(["Paris"], "London")


def test_local_search_uses_searchr1_em_without_external_judge():
    env = LocalSearch.__new__(LocalSearch)
    env.reward_mode = "searchr1_em"
    env.answer_aliases = ["The Author", "Author"]
    env.label_answer = "The Author"
    env.question = "Who wrote it?"
    audit = []
    assert asyncio.run(env.score_answer("author", audit_sink=audit)) == 1
    assert audit[0]["judge_method"] == "searchr1_em"
    assert audit[0]["judge_model"] is None


def test_wiki18_contents_are_exposed_as_title_and_passage():
    assert split_contents('"Pride and Prejudice"\nA novel by Jane Austen.') == (
        "Pride and Prejudice",
        "A novel by Jane Austen.",
    )
    assert truncate_words("one two three four", 3).startswith("one two three\n")


def test_wiki18_server_batches_concurrent_searches():
    class FakeRetriever:
        def __init__(self):
            self.calls = []

        def search_batch(self, queries, topk):
            self.calls.append((queries, topk))
            return [[{"docid": query, "score": 1.0}] * topk for query in queries]

    async def run():
        retriever = FakeRetriever()
        engine = BatchedSearchEngine(retriever, max_batch_size=8, timeout_ms=20)
        await engine.start()
        try:
            results = await asyncio.gather(engine.submit("first", 2), engine.submit("second", 3))
        finally:
            await engine.stop()
        return retriever.calls, results

    calls, results = asyncio.run(run())
    assert calls == [(["first", "second"], 3)]
    assert [len(result) for result in results] == [2, 3]


def test_faiss_index_conversion_is_chunked_and_float16(tmp_path, monkeypatch):
    vectors = np.arange(24, dtype=np.float32).reshape(6, 4)

    class FakeIndex:
        ntotal = 6
        d = 4

        def reconstruct_n(self, start, count):
            return vectors[start : start + count]

    monkeypatch.setitem(sys.modules, "faiss", SimpleNamespace(read_index=lambda _: FakeIndex()))
    output = tmp_path / "embeddings.npy"
    convert(tmp_path / "input.index", output, chunk_rows=2)
    restored = np.load(output)
    assert restored.dtype == np.float16
    np.testing.assert_array_equal(restored, vectors.astype(np.float16))


def test_nq_hotpot_wrapper_uses_wiki18_without_skillrl_checkout():
    root = Path(__file__).resolve().parents[1]
    download = (root / "scripts/download_nq_hotpot_search_data_vista.sh").read_text(encoding="utf-8")
    launcher = (root / "scripts/train_nq_hotpot_grpo_qwen25_7b_4node_idev.sh").read_text(encoding="utf-8")
    baseline = (root / "scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh").read_text(encoding="utf-8")

    assert "PeterJinGo/wiki-18-e5-index" in download
    assert "PeterJinGo/wiki-18-corpus" in download
    assert "intfloat/e5-base-v2" in download
    assert "convert_faiss_index_to_numpy.py" in download
    assert "git clone" not in download
    assert "wiki18_search_server.py" in launcher
    assert "--embedding-path" in launcher
    assert "conda activate cxtgraph" in launcher
    assert "--gpus-per-node" not in launcher
    assert 'TRAINER_VAL_ONLY:-False' in launcher
    assert 'export WORKFLOW_OVERRIDE=searchr1' in launcher
    assert 'export SEARCH_TOPK_CAP=${SEARCH_TOPK_CAP:-3}' in launcher
    assert launcher.count('--require-all-benchmarks') == 2
    assert '--require-training-health "$RUN_LOG"' in launcher
    assert 'export EXTERNAL_SEARCH_URL="$SEARCH_URL"' in launcher
    assert "EXTERNAL_SEARCH_URL=${EXTERNAL_SEARCH_URL:-}" in baseline
    assert 'WORKFLOW_OVERRIDE=${WORKFLOW_OVERRIDE:-}' in baseline
    assert 'plugin.workflow_override=$WORKFLOW_OVERRIDE' in baseline


def test_converted_rows_survive_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    converted, _ = convert_frame(_raw_frame(), "train", seed=7)
    path = tmp_path / "train.parquet"
    converted.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    row = restored.loc[restored.data_source == "searchR1_nq"].iloc[0]
    assert row.extra_info["query"] == "Who wrote it?"
    assert list(row.extra_info["answer_aliases"]) == ["The Author", "Author"]
    assert row.extra_info["workflow"] == "searchr1"
