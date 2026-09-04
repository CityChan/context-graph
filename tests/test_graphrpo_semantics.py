import asyncio
from types import SimpleNamespace

import numpy as np
import pytest


def _torch():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "zeros"):
        pytest.skip("a complete PyTorch installation is unavailable")
    return torch


def _terminal_rewards(values: list[float], width: int = 4):
    torch = _torch()
    rewards = torch.zeros((len(values), width), dtype=torch.float32)
    rewards[:, -1] = torch.tensor(values, dtype=torch.float32)
    return rewards


def test_graphrpo_deduplicates_streams_and_uses_population_std():
    torch = _torch()
    from verl.trainer.ppo.core_algos import compute_graphrpo_advantage

    mask = torch.ones((3, 4), dtype=torch.float32)
    advantages, _ = compute_graphrpo_advantage(
        token_level_rewards=_terminal_rewards([0.0, 0.0, 1.0]),
        response_mask=mask,
        index=np.array(["q", "q", "q"], dtype=object),
        gen_uid=np.array(["episode-0", "episode-0", "episode-1"], dtype=object),
        config={"graphrpo_alpha": 1.0, "graphrpo_beta": 1.0},
    )

    # Deduplicated rewards are [0, 1]. Their population std is 0.5, so
    # the two episode advantages are exactly -1 and +1.
    assert torch.allclose(advantages[0], torch.full((4,), -1.0))
    assert torch.allclose(advantages[1], torch.full((4,), -1.0))
    assert torch.allclose(advantages[2], torch.full((4,), 1.0))


def test_graphrpo_adds_edit_and_process_credit_after_group_normalization():
    torch = _torch()
    from verl.trainer.ppo.core_algos import compute_graphrpo_advantage

    mask = torch.ones((2, 4), dtype=torch.float32)
    edit = torch.zeros_like(mask)
    edit[1, 1:3] = 0.4
    process = torch.zeros_like(mask)
    process[1, 2:] = -0.3
    advantages, _ = compute_graphrpo_advantage(
        token_level_rewards=_terminal_rewards([0.0, 1.0]),
        response_mask=mask,
        index=np.array(["q", "q"], dtype=object),
        gen_uid=np.array(["episode-0", "episode-1"], dtype=object),
        graph_edit_credit_mask=edit,
        process_reward_mask=process,
        config={"graphrpo_alpha": 2.0, "graphrpo_beta": 0.5},
    )

    assert advantages[1].tolist() == pytest.approx([1.0, 1.8, 1.65, 0.85])


def test_graphrpo_zero_variance_keeps_only_local_credit():
    torch = _torch()
    from verl.trainer.ppo.core_algos import compute_graphrpo_advantage

    mask = torch.ones((2, 2), dtype=torch.float32)
    edit = torch.tensor([[0.25, 0.25], [0.0, 0.0]], dtype=torch.float32)
    process = torch.tensor([[0.0, -0.2], [0.0, 0.0]], dtype=torch.float32)
    advantages, _ = compute_graphrpo_advantage(
        token_level_rewards=_terminal_rewards([1.0, 1.0], width=2),
        response_mask=mask,
        index=np.array(["q", "q"], dtype=object),
        gen_uid=np.array(["episode-0", "episode-1"], dtype=object),
        graph_edit_credit_mask=edit,
        process_reward_mask=process,
        config={"graphrpo_alpha": 1.0, "graphrpo_beta": 1.0},
    )

    assert advantages[0].tolist() == pytest.approx([0.25, 0.05])
    assert advantages[1].abs().sum().item() == pytest.approx(0.0)


def test_graphrpo_weights_average_within_episode_then_across_group():
    torch = _torch()
    from verl.trainer.ppo.core_algos import compute_graphrpo_loss_weights

    mask = torch.tensor(
        [
            [1, 1, 0, 0],  # episode-0 main: two policy tokens
            [1, 0, 0, 0],  # episode-0 branch: one policy token
            [1, 0, 0, 0],  # episode-1 main: one policy token
        ],
        dtype=torch.float32,
    )
    weights = compute_graphrpo_loss_weights(
        mask,
        np.array(["q", "q", "q"], dtype=object),
        np.array(["episode-0", "episode-0", "episode-1"], dtype=object),
    )

    assert weights[0, :2].tolist() == pytest.approx([1 / 6, 1 / 6])
    assert weights[1, 0].item() == pytest.approx(1 / 6)
    assert weights[2, 0].item() == pytest.approx(1 / 2)
    assert weights[:2].sum().item() == pytest.approx(0.5)
    assert weights[2].sum().item() == pytest.approx(0.5)
    assert weights.sum().item() == pytest.approx(1.0)


def test_graphrpo_rejects_empty_episode_and_nonbinary_reward():
    torch = _torch()
    from verl.trainer.ppo.core_algos import (
        compute_graphrpo_advantage,
        compute_graphrpo_loss_weights,
    )

    with pytest.raises(ValueError, match="M_g=empty"):
        compute_graphrpo_loss_weights(
            torch.tensor([[1, 0], [0, 0]], dtype=torch.float32),
            np.array(["q", "q"], dtype=object),
            np.array(["episode-0", "episode-1"], dtype=object),
        )
    with pytest.raises(ValueError, match="binary task rewards"):
        compute_graphrpo_advantage(
            token_level_rewards=_terminal_rewards([0.25, 1.0], width=2),
            response_mask=torch.ones((2, 2), dtype=torch.float32),
            index=np.array(["q", "q"], dtype=object),
            gen_uid=np.array(["episode-0", "episode-1"], dtype=object),
        )


def test_graphrpo_policy_loss_uses_precomputed_global_weights():
    torch = _torch()
    from verl.trainer.ppo.core_algos import compute_policy_loss_graphrpo

    config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        global_batch_info={"graphrpo_dp_size": 1},
    )
    loss, _ = compute_policy_loss_graphrpo(
        old_log_prob=torch.zeros((1, 2)),
        log_prob=torch.zeros((1, 2), requires_grad=True),
        advantages=torch.tensor([[1.0, 2.0]]),
        response_mask=torch.ones((1, 2)),
        config=config,
        graphrpo_loss_weights=torch.tensor([[0.25, 0.75]]),
    )
    assert loss.item() == pytest.approx(-(0.25 * 1.0 + 0.75 * 2.0))


def test_graph_utility_matches_bounded_logit_and_length_penalty():
    from agents.graph_rpo import graph_utility

    utility = graph_utility(
        probability=0.8,
        serialized_tokens=512,
        budget_tokens=2048,
        probability_epsilon=1e-4,
        confidence_bound=8.0,
        serialization_penalty=0.2,
    )
    assert utility == pytest.approx(np.log(4.0) - 0.05)


def test_smoke_evaluator_is_deterministic_and_bounded():
    from scripts.serve_graph_evaluator_smoke import deterministic_probability

    first = deterministic_probability("question", "graph-a")
    assert first == deterministic_probability("question", "graph-a")
    assert first != deterministic_probability("question", "graph-b")
    assert 0.2 <= first <= 0.8


def test_graph_edit_credit_only_uses_valid_state_changing_edits(monkeypatch):
    import agents.graph_rpo as graph_rpo

    async def fake_score(**kwargs):
        assert kwargs["graph_views"] == ["before", "after"]
        return [0.25, 0.75]

    monkeypatch.setattr(graph_rpo, "score_graph_views", fake_score)

    class FakeAgent:
        def __init__(self):
            self.credits = {}

        def add_graph_edit_credit(self, turn, credit):
            self.credits[turn] = self.credits.get(turn, 0.0) + credit

    trace = {
        "events": [
            {
                "seq": 0,
                "source": "model",
                "success": True,
                "op": "merge",
                "before_hash": "a",
                "after_hash": "b",
                "rendered_before": "before",
                "rendered_after": "after",
                "assistant_turn_index": 3,
            },
            {
                "seq": 1,
                "source": "model",
                "success": True,
                "op": "pass",
                "before_hash": "b",
                "after_hash": "b",
                "rendered_before": "after",
                "rendered_after": "after",
                "assistant_turn_index": 5,
            },
        ]
    }
    agent = FakeAgent()
    metrics = asyncio.run(
        graph_rpo.assign_graph_edit_credits(
            agent=agent,
            graph_trace=trace,
            question="question",
            terminal_reward=1.0,
            tokenizer=SimpleNamespace(encode=lambda text, **_: text.split()),
            plugin_config={
                "graph_rpo_evaluator_url": "http://evaluator.invalid/score",
                "graph_rpo_operation_costs": {"merge": 0.1},
                "graph_rpo_delta_max": 1.0,
            },
        )
    )

    assert agent.credits == {3: pytest.approx(1.0)}
    assert trace["events"][0]["graph_rpo_delta_unclipped"] > 1.0
    assert trace["events"][0]["graph_rpo_delta"] == pytest.approx(1.0)
    assert "graph_rpo_delta" not in trace["events"][1]
    assert metrics["graph_rpo_valid_edits"] == 1


def test_failed_episode_outcome_gates_graph_credit_without_evaluator(monkeypatch):
    import agents.graph_rpo as graph_rpo

    async def should_not_run(**_):
        raise AssertionError("the evaluator must not run for R=0")

    monkeypatch.setattr(graph_rpo, "score_graph_views", should_not_run)

    class FakeAgent:
        def add_graph_edit_credit(self, *_):
            raise AssertionError("a failed episode must receive zero edit credit")

    trace = {
        "events": [
            {
                "seq": 0,
                "source": "model",
                "success": True,
                "op": "select",
                "before_hash": "a",
                "after_hash": "b",
            }
        ]
    }
    metrics = asyncio.run(
        graph_rpo.assign_graph_edit_credits(
            agent=FakeAgent(),
            graph_trace=trace,
            question="question",
            terminal_reward=0.0,
            tokenizer=None,
            plugin_config={},
        )
    )
    assert trace["events"][0]["graph_rpo_delta"] == 0.0
    assert trace["events"][0]["graph_rpo_outcome_gated"] is True
    assert metrics["graph_rpo_scored_states"] == 0


def test_graph_evaluator_rows_use_trace_root_and_binary_outcome():
    from scripts.prepare_graph_evaluator_data import result_rows

    result = {
        "task_id": "task-1",
        "status": "success",
        "task_reward": 1.0,
        "graph_state": "after",
        "graph_trace": {
            "initial_graph": {
                "root_id": "n1",
                "nodes": [{"id": "n1", "content": "Who is the answer?"}],
            },
            "events": [
                {"rendered_before": "before", "rendered_after": "after"},
                {"rendered_before": "after", "rendered_after": "final"},
            ],
        },
    }
    rows = result_rows(result)
    assert {row["graph_view"] for row in rows} == {"before", "after", "final"}
    assert {row["question"] for row in rows} == {"Who is the answer?"}
    assert {row["label"] for row in rows} == {1}


def test_graph_evaluator_loader_accepts_verl_jsonl(tmp_path):
    import json

    from scripts.prepare_graph_evaluator_data import load_results

    path = tmp_path / "1.jsonl"
    records = [{"task_id": "a"}, {"task_id": "b"}]
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    assert load_results(path) == records

    single_path = tmp_path / "single.jsonl"
    single_path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    assert load_results(single_path) == records[:1]


def test_relaxed_em_cannot_create_positive_task_reward(monkeypatch):
    from envs import local_search

    label = "Particle Film Application Influences Apple Leaf Physiology, Fruit Yield, and Fruit Quality"
    prediction = (
        'The article titled "Influence of Sunlight Incidence and Fruit Chemical Features '
        "on Oviposition Site Selection in Mango by Anastrepha obliqua: Implications for Management\""
    )
    assert local_search.em_score(label, prediction) is False
    assert local_search.relaxed_em(label, prediction) is True

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    audit = []
    score = asyncio.run(local_search.judge("question", label, prediction, audit_sink=audit))
    assert score == 0
    assert audit[0]["judge_method"] == "offline_strict_only"
    assert audit[0]["relaxed_em"] is True


def test_negative_llm_judgment_is_not_overridden_by_relaxed_em(monkeypatch):
    from envs import local_search

    async def negative_judge(*_, **__):
        return "extracted_final_answer: Josef Sommer\nreasoning: Missing the full identity.\ncorrect: no\nconfidence: 100"

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-5-nano")
    monkeypatch.setattr(local_search, "call_openai_raw", negative_judge)
    audit = []
    score = asyncio.run(
        local_search.judge(
            "question",
            "Maximilian Josef Sommer",
            "Josef Sommer",
            audit_sink=audit,
        )
    )
    assert local_search.relaxed_em("Maximilian Josef Sommer", "Josef Sommer") is True
    assert score == 0
    assert audit[0]["judge_method"] == "llm_judge"
    assert audit[0]["grader_attempts"][0]["parsed"]["correct"] is False


def test_judge_audit_deduplicates_branch_streams_and_checks_reward(tmp_path):
    import json

    from scripts.audit_bc_judge_results import audit_results

    audit = {
        "correct_answer": "Mukul Pal",
        "predicted_answer": "Mukul Pal",
        "strict_em": True,
        "relaxed_em": True,
        "judge_method": "strict_em",
        "score": 1,
    }
    records = [
        {"task_id": "task", "gen_uid": "episode", "task_reward": 1, "judge_audit": [audit]},
        {"task_id": "task", "gen_uid": "episode", "task_reward": 1, "judge_audit": [audit]},
    ]
    path = tmp_path / "1.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    report = audit_results([path])
    assert report["summary"]["records"] == 2
    assert report["summary"]["unique_judge_decisions"] == 1
    assert report["summary"]["positive_decisions"] == 1
    assert report["summary"]["task_reward_audit_mismatches"] == 0


def test_env_stats_metrics_are_episode_weighted_not_branch_weighted():
    _torch()
    from verl.workers.reward_manager.agent import AgentLoopRewardManager

    data = type(
        "Batch",
        (),
        {
            "non_tensor_batch": {
                "gen_uid": np.array(["episode-a", "episode-a", "episode-b"], dtype=object),
                "env_stats": np.array([
                    {"task_reward": 1.0, "graph_rpo_valid_edits": 2.0},
                    {"task_reward": 1.0, "graph_rpo_valid_edits": 2.0},
                    {"task_reward": 0.0, "graph_rpo_valid_edits": 0.0},
                ], dtype=object),
            },
            "meta_info": {},
            "__len__": lambda self: 3,
        },
    )()
    metrics = AgentLoopRewardManager.__new__(AgentLoopRewardManager)._compute_batch_metrics(
        data, [1.0, 1.0, 0.0]
    )
    assert metrics["task_reward"].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert metrics["graph_rpo_valid_edits"].tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_graphrpo_training_wiring_is_explicit():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    trainer = (root / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8")
    actor = (root / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8")
    agent = (root / "agents/graph_agent_isolated.py").read_text(encoding="utf-8")
    agent_utils = (root / "agents/utils.py").read_text(encoding="utf-8")
    reward_manager = (
        root / "verl/workers/reward_manager/agent.py"
    ).read_text(encoding="utf-8")
    launcher = (root / "scripts/train_bc_ctxgraph_8b_graphrpo_5node_48h.sh").read_text(
        encoding="utf-8"
    )
    smoke_launcher = (
        root / "scripts/smoke_train_bc_ctxgraph_8b_graphrpo_5node_idev.sh"
    ).read_text(encoding="utf-8")

    assert "AdvantageEstimator.GRAPHRPO" in trainer
    assert 'loss_mode == "graphrpo"' in actor
    assert '"graphrpo_loss_weights"' in actor
    assert 'trajectory_fields["task_reward"]' in trainer
    assert "assign_graph_edit_credits" in agent
    assert "process_reward_min_precedence=graph_rpo_enabled" in agent
    assert "self.process_reward_min_precedence" in agent_utils
    assert "not graph_rpo_enabled and process_reward and 'graph' in process_reward" in agent
    assert '"graph_rpo_valid_edits"' in reward_manager
    assert '"graph_rpo_scored_states"' in reward_manager
    assert '"graph_rpo_delta_abs_sum"' in reward_manager
    for metric in (
        "graph_compactness",
        "graph_structural",
        "graph_merge_bonus",
        "graph_prune_bonus",
        "graph_uniqueness_bonus",
        "graph_cost_penalty",
        "graph_invalid_op_penalty",
        "graph_n_folded",
        "graph_n_pruned",
        "graph_n_cross_edges",
        "graph_trace_events",
        "graph_trace_model_events",
    ):
        assert f'"{metric}"' in reward_manager
        assert f"'{metric}'" in agent
    local_search = (root / "envs/local_search.py").read_text(encoding="utf-8")
    for metric in ("judge_relaxed_only", "judge_parse_failure"):
        assert f'"{metric}"' in reward_manager
        assert f'self.stats["{metric}"]' in local_search
    assert "export ADV_ESTIMATOR=graphrpo" in launcher
    assert "export POLICY_LOSS_MODE=graphrpo" in launcher
    assert "serve_graph_evaluator_smoke.py" in smoke_launcher
    assert "global_step_174" in smoke_launcher
    assert "python -m verl.model_merger merge --backend fsdp" in smoke_launcher
    assert 'find -L "$MODEL_PATH"' in smoke_launcher
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-${#NODELIST[@]}}" in smoke_launcher
    assert "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$TRAINER_NODES}" in smoke_launcher
    assert "TOTAL_TRAINING_STEPS:-1" in smoke_launcher
    assert "ROLLOUT_N:-2" in smoke_launcher
    assert "VAL_BEFORE_TRAIN:-False" in smoke_launcher
    assert "SAVE_ROLLOUT_DATA:-1" in smoke_launcher

    base_launcher = (
        root / "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh"
    ).read_text(encoding="utf-8")
    assert '$NUM_NODES nodes [1 search + $((NUM_NODES - 1)) trainer]' in base_launcher
    assert 'trainer.rollout_data_dir=$ROLLOUT_DATA_DIR' in base_launcher
