from scripts.prepare_miroverse_search_policy_seeds import build_seeds, record_to_seed


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
    assert seed["reward_model"]["ground_truth"] == "Example Person"


def test_requires_boxed_answer_and_deduplicates_queries():
    rejected = source_record("No answer?")
    rejected["messages"][-3]["content"] = "I do not know."
    reasons = []
    assert record_to_seed(rejected, 0, reasons) is None
    assert reasons == ["missing_boxed_answer"]
    seeds, counters = build_seeds([source_record(), source_record()], 0)
    assert len(seeds) == 1
    assert counters["accepted"] == 1
    assert counters["duplicates"] == 1
