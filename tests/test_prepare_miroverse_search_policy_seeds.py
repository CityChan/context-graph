from pathlib import Path

from scripts.prepare_miroverse_search_policy_seeds import (
    build_seeds,
    normalize_miroverse_answer,
    record_to_seed,
)


def source_record(question="Who played the role?"):
    return {
        "messages": [
            {"role": "system", "content": "tools"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "tool reasoning"},
            {"role": "user", "content": "evidence"},
            {"role": "assistant", "content": "Evidence supports \\boxed{Example Person}."},
            {"role": "user", "content": "Summarize the above conversation and output the FINAL ANSWER."},
            {"role": "assistant", "content": "\\boxed{Example Person}"},
        ]
    }


def test_extracts_query_and_exact_answer_before_artificial_finalizer():
    seed = record_to_seed(source_record(), 7)
    assert seed is not None
    assert seed["extra_info"]["query"] == "Who played the role?"
    assert seed["extra_info"]["answer"] == "Example Person"
    assert seed["extra_info"]["workflow"] == "search_graph"
    assert seed["ability"] == "LocalSearch"
    assert seed["data_source"] == "miroverse_musique"
    assert seed["reward_model"]["ground_truth"] == "Example Person"


def test_uses_boxed_finalizer_as_gold_when_candidate_answer_is_plain():
    record = source_record()
    record["messages"][-3]["content"] = "The answer is Example Person."
    seed = record_to_seed(record, 7)
    assert seed is not None
    assert seed["reward_model"]["ground_truth"] == "Example Person"


def test_seed_ability_routes_to_local_search_environment():
    seed = record_to_seed(source_record(), 7)
    selector = Path("agents/utils.py").read_text(encoding="utf-8")
    assert seed["ability"] == "LocalSearch"
    assert "'LocalSearch' in ability" in selector


def test_requires_boxed_answer_and_deduplicates_queries():
    rejected = source_record("No answer?")
    rejected["messages"][-3]["content"] = "I do not know."
    rejected["messages"][-1]["content"] = "I do not know."
    reasons = []
    assert record_to_seed(rejected, 0, reasons) is None
    assert reasons == ["missing_boxed_answer"]
    seeds, counters = build_seeds([source_record(), source_record()], 0)
    assert len(seeds) == 1
    assert counters["accepted"] == 1
    assert counters["duplicates"] == 1


def test_normalizes_textual_latex_gold_answers_for_exact_match_reward():
    assert normalize_miroverse_answer(r'Dorothy\ "Dottie"\ Hinson') == 'Dorothy "Dottie" Hinson'
    assert normalize_miroverse_answer(r"June\ 10,\ 323\ BC") == "June 10, 323 BC"
    assert normalize_miroverse_answer(r"\text{Greyhound Canada}") == "Greyhound Canada"
