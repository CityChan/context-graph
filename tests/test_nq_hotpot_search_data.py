import asyncio
import pandas as pd
import pytest

from envs.local_search import LocalSearch, searchr1_em_score
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


def test_converted_rows_survive_parquet_round_trip(tmp_path):
    pytest.importorskip("pyarrow")
    converted, _ = convert_frame(_raw_frame(), "train", seed=7)
    path = tmp_path / "train.parquet"
    converted.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    row = restored.loc[restored.data_source == "searchR1_nq"].iloc[0]
    assert row.extra_info["query"] == "Who wrote it?"
    assert list(row.extra_info["answer_aliases"]) == ["The Author", "Author"]
