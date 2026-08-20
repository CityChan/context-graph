from argparse import Namespace

from scripts.build_contextgraph_sft import build_rows, result_to_sft_row


def _result(**overrides):
    result = {
        "task_id": "gaia-1",
        "status": "success",
        "task_reward": 1.0,
        "is_finish": True,
        "messages": [
            {"role": "system", "content": "Use ContextGraph."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<function=prune><parameter=node_id>n2</parameter></function>"},
            {"role": "user", "content": "Pruned n2."},
            {"role": "assistant", "content": "<function=finish><parameter=answer>A</parameter></function>"},
        ],
        "env_stats": {
            "graph_explicit_ops": 1,
            "graph_invalid_ops": 0,
            "graph_n_nodes": 3,
            "graph_n_edges": 2,
            "overlong": 0,
            "hit_token_limit": 0,
        },
    }
    result.update(overrides)
    return result


def test_result_to_sft_row_accepts_verified_success():
    row = result_to_sft_row(
        _result(),
        min_task_reward=1.0,
        min_valid_graph_ops=1,
        max_invalid_graph_ops=0,
        require_finish=True,
        enable_thinking=False,
        min_structural_graph_ops=1,
        teacher_model="deepseek-ai/DeepSeek-V4-Flash-0731",
    )
    assert row is not None
    assert row["graph_valid_ops"] == 1
    assert row["graph_invalid_ops"] == 0
    assert row["enable_thinking"] is False
    assert row["graph_structural_ops"] == 1
    assert row["teacher_model"] == "deepseek-ai/DeepSeek-V4-Flash-0731"


def test_result_to_sft_row_rejects_invalid_graph_operation():
    result = _result()
    result["env_stats"]["graph_invalid_ops"] = 1
    assert result_to_sft_row(
        result,
        min_task_reward=1.0,
        min_valid_graph_ops=1,
        max_invalid_graph_ops=0,
        require_finish=True,
        enable_thinking=False,
    ) is None


def test_build_rows_deduplicates_identical_conversations():
    args = Namespace(
        min_task_reward=1.0,
        min_valid_graph_ops=1,
        max_invalid_graph_ops=0,
        allow_unfinished=False,
        enable_thinking=False,
    )
    rows, counters = build_rows([_result(), _result(task_id="gaia-duplicate")], args)
    assert len(rows) == 1
    assert counters == {"input": 2, "accepted": 1, "duplicates": 1, "rejected": 0}


def test_result_to_sft_row_can_require_structural_graph_operation():
    result = _result(messages=[
        {"role": "system", "content": "Use ContextGraph."},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "<function=select><parameter=node_id>n2</parameter></function>"},
        {"role": "user", "content": "Focus shifted to n2."},
        {"role": "assistant", "content": "<function=finish><parameter=answer>A</parameter></function>"},
    ])
    assert result_to_sft_row(
        result,
        min_task_reward=1.0,
        min_valid_graph_ops=1,
        max_invalid_graph_ops=0,
        require_finish=True,
        enable_thinking=False,
        min_structural_graph_ops=1,
    ) is None
