import asyncio
import importlib.util
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_rollout_json_serializer_handles_numpy_and_metadata_types():
    module_path = Path(__file__).parents[1] / "verl" / "utils" / "json_serialization.py"
    spec = importlib.util.spec_from_file_location("rollout_json_serialization", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    identifier = uuid.uuid4()
    payload = {
        "array": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "float": np.float32(0.25),
        "integer": np.int64(7),
        "path": Path("rollouts/1.jsonl"),
        "uid": identifier,
        "tags": {"judge", "graph"},
    }

    decoded = json.loads(json.dumps(payload, default=module.json_default))
    assert decoded["array"] == [[1, 2], [3, 4]]
    assert decoded["float"] == pytest.approx(0.25)
    assert decoded["integer"] == 7
    assert Path(decoded["path"]) == Path("rollouts/1.jsonl")
    assert decoded["uid"] == str(identifier)
    assert sorted(decoded["tags"]) == ["graph", "judge"]


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


def test_graphrpo_scalar_reward_ablation_and_zero_local_credit():
    torch = _torch()
    from verl.trainer.ppo.core_algos import compute_graphrpo_advantage

    mask = torch.ones((2, 3), dtype=torch.float32)
    edit = torch.tensor([[0.5, -0.5, 0.25], [0.1, 0.2, 0.3]], dtype=torch.float32)
    process = torch.tensor([[0.0, -1.0, -0.5], [-0.2, 0.0, -0.7]], dtype=torch.float32)
    advantages, _ = compute_graphrpo_advantage(
        token_level_rewards=_terminal_rewards([0.2, 2.2], width=3),
        response_mask=mask,
        index=np.array(["q", "q"], dtype=object),
        gen_uid=np.array(["episode-0", "episode-1"], dtype=object),
        graph_edit_credit_mask=edit,
        process_reward_mask=process,
        config={
            "graphrpo_alpha": 0.0,
            "graphrpo_beta": 0.0,
            "graphrpo_require_binary_reward": False,
        },
    )

    assert torch.allclose(advantages[0], torch.full((3,), -1.0))
    assert torch.allclose(advantages[1], torch.full((3,), 1.0))


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
                "graph_rpo_delta_scale": 2.0,
                "graph_rpo_delta_max": 1.0,
            },
        )
    )

    assert agent.credits == {3: pytest.approx(1.0)}
    assert trace["events"][0]["graph_rpo_delta_unclipped"] > 1.0
    assert trace["events"][0]["graph_rpo_delta_scale"] == pytest.approx(2.0)
    assert trace["events"][0]["graph_rpo_delta_scaled_unclipped"] > 1.0
    assert trace["events"][0]["graph_rpo_delta"] == pytest.approx(1.0)
    assert "graph_rpo_delta" not in trace["events"][1]
    assert metrics["graph_rpo_valid_edits"] == 1
    assert metrics["graph_rpo_creditable_edits"] == 1
    assert metrics["graph_rpo_credited_edits"] == 1


def test_graph_edit_credit_ignores_counter_only_legacy_state_changes():
    from agents.graph_rpo import valid_graph_edit_events

    before = {
        "nodes": [{"id": "n0", "status": "active"}],
        "edges": [],
        "root_id": "n0",
        "active_node_id": "n0",
        "counters": {"operation_count": 1},
    }
    after = {
        **before,
        "counters": {"operation_count": 2},
    }
    event = {
        "source": "model",
        "success": True,
        "op": "select",
        "before_hash": "full-before",
        "after_hash": "full-after",
        "before_state": before,
        "after_state": after,
    }

    assert valid_graph_edit_events({"events": [event]}) == []


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


def test_reference_graph_requests_are_success_gated_and_auditable():
    from agents.graph_rpo import prepare_reference_graph_edit_requests

    event = {
        "seq": 4,
        "source": "model",
        "success": True,
        "op": "merge",
        "before_hash": "before-hash",
        "after_hash": "after-hash",
        "rendered_before": "before graph",
        "rendered_after": "after graph",
        "assistant_turn_index": 7,
    }
    requests, metrics = prepare_reference_graph_edit_requests(
        graph_trace={"events": [event]},
        terminal_reward=1.0,
    )
    assert requests == [{"seq": 4, "assistant_turn_index": 7}]
    assert metrics["graph_rpo_valid_edits"] == 1
    assert event["graph_rpo_credit_backend"] == "reference_answer_likelihood"
    assert event["graph_rpo_outcome_gated"] is False

    failed_event = dict(event)
    failed_event.pop("graph_rpo_delta", None)
    requests, _ = prepare_reference_graph_edit_requests(
        graph_trace={"events": [failed_event]},
        terminal_reward=0.0,
    )
    assert requests == []
    assert failed_event["graph_rpo_delta"] == 0.0
    assert failed_event["graph_rpo_outcome_gated"] is True


def test_old_policy_graph_requests_record_selected_backend():
    from agents.graph_rpo import prepare_reference_graph_edit_requests

    event = {
        "seq": 5,
        "source": "model",
        "success": True,
        "op": "prune",
        "before_hash": "before-hash",
        "after_hash": "after-hash",
        "rendered_before": "before graph",
        "rendered_after": "after graph",
        "assistant_turn_index": 8,
    }
    requests, _ = prepare_reference_graph_edit_requests(
        graph_trace={"events": [event]},
        terminal_reward=1.0,
        credit_backend="old_policy_answer_likelihood",
    )
    assert requests == [{"seq": 5, "assistant_turn_index": 8}]
    assert event["graph_rpo_credit_backend"] == "old_policy_answer_likelihood"


def test_counterfactual_qa_credits_paired_task_outcomes_without_success_gate():
    from agents.graph_rpo import assign_counterfactual_graph_edit_credits

    calls = []

    async def generate_answer(view, sample_index, seed):
        calls.append((view, sample_index, seed))
        answer = "correct" if view == "after graph" else "wrong"
        return f"reasoning\n<answer>{answer}</answer>"

    async def score_answer(answer, audit_sink):
        reward = float(answer == "correct")
        audit_sink.append({"score": reward, "judge_method": "test"})
        return reward

    class FakeAgent:
        def __init__(self):
            self.credits = {}

        def add_graph_edit_credit(self, turn, credit):
            self.credits[turn] = self.credits.get(turn, 0.0) + credit

    event = {
        "seq": 9,
        "source": "model",
        "success": True,
        "op": "merge",
        "before_hash": "a",
        "after_hash": "b",
        "rendered_before": "before graph",
        "rendered_after": "after graph",
        "assistant_turn_index": 12,
    }
    agent = FakeAgent()
    metrics = asyncio.run(
        assign_counterfactual_graph_edit_credits(
            agent=agent,
            graph_trace={"events": [event]},
            question="question",
            generate_answer=generate_answer,
            score_answer=score_answer,
            plugin_config={
                "graph_rpo_counterfactual_samples": 2,
                "graph_rpo_operation_costs": {"merge": 0.1},
                "graph_rpo_delta_scale": 3.0,
                "graph_rpo_delta_max": 1.0,
                "graph_rpo_counterfactual_seed": 7,
            },
        )
    )

    assert agent.credits == {12: pytest.approx(0.3)}
    assert [call[2] for call in calls[:2]] == [call[2] for call in calls[2:]]
    assert event["graph_rpo_credit_backend"] == "old_policy_counterfactual_qa"
    assert event["graph_rpo_outcome_gated"] is False
    assert len(event["graph_rpo_counterfactual_seeds"]) == 2
    assert event["graph_rpo_utility_before"] == pytest.approx(0.0)
    assert event["graph_rpo_utility_after"] == pytest.approx(1.0)
    assert event["graph_rpo_delta_unclipped"] == pytest.approx(0.9)
    assert event["graph_rpo_delta_scale"] == pytest.approx(3.0)
    assert event["graph_rpo_delta_scaled_unclipped"] == pytest.approx(0.3)
    assert len(event["graph_rpo_counterfactual_before_responses"]) == 2
    assert len(event["graph_rpo_counterfactual_after_responses"]) == 2
    assert event["graph_rpo_counterfactual_before_rewards"] == [0.0, 0.0]
    assert event["graph_rpo_counterfactual_after_rewards"] == [1.0, 1.0]
    assert metrics["graph_rpo_creditable_edits"] == 1
    assert metrics["graph_rpo_credited_edits"] == 1
    assert metrics["graph_rpo_scored_states"] == 2
    assert metrics["graph_rpo_delta_abs_sum"] == pytest.approx(0.3)
    assert metrics["graph_rpo_counterfactual_scored_states"] == 2
    assert metrics["graph_rpo_counterfactual_probe_rollouts"] == 4
    assert metrics["graph_rpo_counterfactual_tagged_responses"] == 4
    assert metrics["graph_rpo_counterfactual_tag_rate"] == 1.0
    assert metrics["graph_rpo_counterfactual_positive_rewards"] == 2
    assert metrics["graph_rpo_counterfactual_positive_rate"] == 0.5
    assert metrics["graph_rpo_counterfactual_nonzero_edits"] == 1
    assert metrics["graph_rpo_counterfactual_delta_abs_sum"] == pytest.approx(0.3)


def test_counterfactual_qa_prompt_and_answer_extraction():
    from agents.graph_rpo import (
        extract_counterfactual_answer,
        format_counterfactual_qa_messages,
        graph_rpo_credit_backend,
    )

    assert graph_rpo_credit_backend({}) == "old_policy_counterfactual_qa"
    messages = format_counterfactual_qa_messages("Who?", "[n1] evidence")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Do not search or call tools" in messages[0]["content"]
    assert "Who?" in messages[1]["content"]
    assert "[n1] evidence" in messages[1]["content"]
    assert extract_counterfactual_answer("x <answer>first</answer> <answer>last</answer>") == "last"
    assert extract_counterfactual_answer("Final answer: fallback.") == "fallback"


def test_counterfactual_qa_does_not_judge_malformed_probe_output():
    from agents.graph_rpo import assign_counterfactual_graph_edit_credits

    score_calls = []

    async def generate_answer(view, sample_index, seed):
        return "unfinished reasoning without a submitted answer"

    async def score_answer(answer, audit_sink):
        score_calls.append(answer)
        return 1.0

    class FakeAgent:
        def add_graph_edit_credit(self, turn, credit):
            self.credit = credit

    event = {
        "seq": 10,
        "source": "model",
        "success": True,
        "op": "merge",
        "before_hash": "a",
        "after_hash": "b",
        "rendered_before": "before graph",
        "rendered_after": "after graph",
        "assistant_turn_index": 4,
    }
    agent = FakeAgent()
    metrics = asyncio.run(
        assign_counterfactual_graph_edit_credits(
            agent=agent,
            graph_trace={"events": [event]},
            question="question",
            generate_answer=generate_answer,
            score_answer=score_answer,
            plugin_config={"graph_rpo_counterfactual_samples": 1},
        )
    )

    assert score_calls == []
    assert agent.credit == 0.0
    assert metrics["graph_rpo_credited_edits"] == 0
    assert metrics["graph_rpo_counterfactual_tag_rate"] == 0.0
    assert metrics["graph_rpo_counterfactual_positive_rewards"] == 0
    assert event["graph_rpo_counterfactual_before_judge_audits"][0][0][
        "judge_method"
    ] == "counterfactual_format_invalid"


def test_local_search_can_score_counterfactual_answer_without_mutating_episode():
    from envs.local_search import LocalSearch

    env = LocalSearch.__new__(LocalSearch)
    env.question = "Who?"
    env.label_answer = "Ada Lovelace"
    env.predicted_answer = ("wrong main answer", "", 0.0)
    audit = []
    reward = asyncio.run(env.score_answer("Ada Lovelace", audit_sink=audit))

    assert reward == 1
    assert env.predicted_answer[0] == "wrong main answer"
    assert audit[-1]["strict_em"] is True


def test_reference_answer_batch_and_mean_likelihood_use_only_answer_tokens():
    torch = _torch()
    from verl.trainer.ppo.reference_graph_credit import (
        ReferenceViewKey,
        build_reference_scoring_batch,
        mean_answer_log_likelihood,
    )

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 9

        def apply_chat_template(self, messages, **_):
            return [11, 12] + [ord(char) % 50 + 1 for char in messages[-1]["content"]]

        def encode(self, text, **_):
            return [ord(char) % 50 + 1 for char in text]

    keys = [
        ReferenceViewKey("question", "AB", "before"),
        ReferenceViewKey("question", "C", "after"),
    ]
    scoring_batch, answer_mask = build_reference_scoring_batch(
        FakeTokenizer(),
        keys,
        max_prompt_length=12,
        max_answer_length=8,
    )
    assert scoring_batch.batch["prompts"].shape == (2, 12)
    assert scoring_batch.batch["responses"].shape == (2, 2)
    assert answer_mask.tolist() == [[1, 1], [1, 0]]
    assert scoring_batch.meta_info["ref_log_prob_temperature"] == 1.0
    assert scoring_batch.meta_info["log_prob_temperature_override"] == 1.0

    likelihoods = mean_answer_log_likelihood(
        torch.tensor([[-2.0, -4.0], [-1.5, -99.0]]),
        answer_mask,
    )
    assert likelihoods == pytest.approx([-3.0, -1.5])


def test_reference_answer_likelihood_delta_maps_to_edit_tokens():
    torch = _torch()
    from verl import DataProto
    from verl.trainer.ppo.reference_graph_credit import (
        apply_reference_edit_credits,
        collect_reference_edit_plans,
    )

    event = {
        "seq": 2,
        "source": "model",
        "success": True,
        "op": "merge",
        "rendered_before": "before graph",
        "rendered_after": "after graph",
    }
    trace = {"events": [event]}
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones((2, 5), dtype=torch.long),
            "response_mask": torch.tensor(
                [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.long
            ),
        },
        non_tensors={
            "agent_name": np.array(["main", "branch-0"], dtype=object),
            "gen_uid": np.array(["episode", "episode"], dtype=object),
            "graph_rpo_reference_edits": np.array(
                [[{"seq": 2, "response_token_indices": [1, 2]}], []], dtype=object
            ),
            "graph_rpo_reference_question": np.array(["question", "question"], dtype=object),
            "graph_rpo_reference_answer": np.array(["answer", "answer"], dtype=object),
            "graph_trace": np.array([trace, {"events": [dict(event)]}], dtype=object),
            "env_stats": np.array([{}, {}], dtype=object),
        },
    )
    plans, keys = collect_reference_edit_plans(batch)
    assert len(plans) == 1
    assert len(keys) == 2
    likelihoods = {
        plans[0].before_key: -2.0,
        plans[0].after_key: -1.4,
    }
    metrics = apply_reference_edit_credits(
        batch,
        plans,
        likelihoods,
        delta_max=1.0,
        delta_scale=2.0,
        operation_costs={"merge": 0.1},
    )
    assert torch.allclose(
        batch.batch["graph_edit_credit_mask"],
        torch.tensor([[0.0, 0.25, 0.25, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert event["graph_rpo_answer_log_likelihood_before"] == pytest.approx(-2.0)
    assert event["graph_rpo_answer_log_likelihood_after"] == pytest.approx(-1.4)
    assert event["graph_rpo_delta_unclipped"] == pytest.approx(0.5)
    assert event["graph_rpo_delta_scale"] == pytest.approx(2.0)
    assert event["graph_rpo_delta_scaled_unclipped"] == pytest.approx(0.25)
    assert event["graph_rpo_delta"] == pytest.approx(0.25)
    assert batch.non_tensor_batch["graph_trace"][1]["events"][0][
        "graph_rpo_delta"
    ] == pytest.approx(0.25)
    assert batch.non_tensor_batch["env_stats"][1]["graph_rpo_scored_states"] == 2
    assert metrics["graphrpo/reference_creditable_edits"] == 1
    assert metrics["graphrpo/reference_scored_states"] == 2
    with pytest.raises(ValueError, match="graph_rpo_delta_scale"):
        apply_reference_edit_credits(
            batch,
            plans,
            likelihoods,
            delta_max=1.0,
            delta_scale=0.0,
        )


def test_old_policy_answer_likelihood_uses_separate_metric_namespace():
    torch = _torch()
    from verl import DataProto
    from verl.trainer.ppo.reference_graph_credit import (
        ReferenceEditPlan,
        ReferenceViewKey,
        apply_reference_edit_credits,
    )

    before_key = ReferenceViewKey("question", "answer", "before")
    after_key = ReferenceViewKey("question", "answer", "after")
    event = {"seq": 1, "op": "add_edge"}
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones((1, 3), dtype=torch.long),
            "response_mask": torch.ones((1, 3), dtype=torch.long),
        }
    )
    plan = ReferenceEditPlan(0, event, before_key, after_key, [0, 1])
    metrics = apply_reference_edit_credits(
        batch,
        [plan],
        {before_key: -2.0, after_key: -1.8},
        delta_max=0.25,
        credit_backend="old_policy_answer_likelihood",
        metric_namespace="old_policy",
    )
    assert event["graph_rpo_credit_backend"] == "old_policy_answer_likelihood"
    assert metrics["graphrpo/old_policy_creditable_edits"] == 1
    assert metrics["graphrpo/old_policy_scored_states"] == 2
    assert metrics["graphrpo/old_policy_delta_sum"] == pytest.approx(0.2)


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


def test_graph_evaluator_ignores_aggregate_task_reward_for_episode_label():
    from scripts.prepare_graph_evaluator_data import result_rows

    result = {
        "task_reward": 0.11139705882352942,
        "score": 1.0,
        "env_stats": {"task_reward": 1.0},
        "graph_state": "final graph",
        "graph_trace": {
            "initial_graph": {
                "root_id": "n1",
                "nodes": [{"id": "n1", "content": "BrowseComp question"}],
            },
            "events": [],
        },
    }
    rows = result_rows(result)
    assert len(rows) == 1
    assert rows[0]["label"] == 1


def test_graph_evaluator_falls_back_to_per_episode_score():
    from scripts.prepare_graph_evaluator_data import result_rows

    result = {
        "task_reward": 0.25,
        "score": 0.0,
        "graph_state": "final graph",
        "graph_trace": {
            "initial_graph": {
                "root_id": "n1",
                "nodes": [{"id": "n1", "content": "Question"}],
            },
            "events": [],
        },
    }
    rows = result_rows(result)
    assert len(rows) == 1
    assert rows[0]["label"] == 0


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


def test_graph_evaluator_preserves_different_episode_outcomes(tmp_path):
    import json

    from scripts.prepare_graph_evaluator_data import build_rows

    trace = {
        "initial_graph": {
            "root_id": "n1",
            "nodes": [{"id": "n1", "content": "Question"}],
        },
        "events": [{"rendered_before": "same", "rendered_after": "same"}],
    }
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": "task",
                    "gen_uid": f"episode-{label}",
                    "question": "Question",
                    "task_reward": label,
                    "graph_trace": trace,
                }
            )
            for label in (0, 1)
        )
        + "\n",
        encoding="utf-8",
    )
    rows = build_rows([path])
    assert len(rows) == 2
    assert {row["label"] for row in rows} == {0, 1}


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


def test_judge_audit_uses_episode_outcome_before_aggregate_metric(tmp_path):
    import json

    from scripts.audit_bc_judge_results import audit_results

    record = {
        "task_reward": 0.11139705882352942,
        "score": 1.0,
        "env_stats": {"task_reward": 1.0},
        "judge_audit": [{"judge_method": "strict_em", "score": 1}],
    }
    path = tmp_path / "aggregate-metric.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = audit_results([path])
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
    audit_smoke_launcher = (
        root / "scripts/smoke_train_bc_ctxgraph_8b_graphrpo_qwen3_8b_4node_judge_audit.sh"
    ).read_text(encoding="utf-8")
    zeroshot_pilot_launcher = (
        root / "scripts/pilot_train_bc_ctxgraph_8b_graphrpo_ref_zeroshot_4node_20step_val.sh"
    ).read_text(encoding="utf-8")
    matched_scale_submitter = (
        root / "scripts/submit_matched_graphrpo_delta_scale_20step.sh"
    ).read_text(encoding="utf-8")
    scale_smoke_submitter = (
        root / "scripts/submit_graphrpo_delta_scale_smoke.sh"
    ).read_text(encoding="utf-8")
    old_policy_smoke_launcher = (
        root / "scripts/smoke_train_bc_ctxgraph_8b_graphrpo_old_policy_4node_2step.sh"
    ).read_text(encoding="utf-8")
    counterfactual_smoke_launcher = (
        root / "scripts/smoke_train_bc_ctxgraph_8b_graphrpo_counterfactual_4node_2step.sh"
    ).read_text(encoding="utf-8")
    counterfactual_stage2_submitter = (
        root / "scripts/submit_stage2_graphrpo_counterfactual.sh"
    ).read_text(encoding="utf-8")
    paperfaithful_launcher = (
        root / "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh"
    ).read_text(encoding="utf-8")
    fsdp_worker = (root / "verl/workers/fsdp_workers.py").read_text(encoding="utf-8")

    assert "AdvantageEstimator.GRAPHRPO" in trainer
    assert 'loss_mode == "graphrpo"' in actor
    assert '"graphrpo_loss_weights"' in actor
    assert 'trajectory_fields["task_reward"]' in trainer
    assert trainer.count('trajectory_fields["task_reward"]') >= 2
    assert trainer.count('"judge_audit", "uid", "gen_uid"') >= 2
    assert "assign_graph_edit_credits" in agent
    assert "assign_counterfactual_graph_edit_credits" in agent
    assert "prepare_reference_graph_edit_requests" in agent
    assert "_compute_answer_likelihood_graph_credit" in trainer
    assert "old_policy_answer_likelihood" in trainer
    assert "old_policy_counterfactual_qa" not in trainer
    assert "compute_answer_log_prob" in trainer
    assert "def compute_answer_log_prob" in fsdp_worker
    assert "process_reward_min_precedence=graph_rpo_enabled" in agent
    assert "self.process_reward_min_precedence" in agent_utils
    assert "not graph_rpo_enabled and process_reward and 'graph' in process_reward" in agent
    assert '"graph_rpo_valid_edits"' in reward_manager
    assert '"graph_rpo_creditable_edits"' in reward_manager
    assert '"graph_rpo_scored_states"' in reward_manager
    assert '"graph_rpo_delta_abs_sum"' in reward_manager
    assert '"graph_rpo_counterfactual_probe_rollouts"' in reward_manager
    assert '"graph_rpo_counterfactual_tag_rate"' in reward_manager
    assert '"graph_rpo_counterfactual_nonzero_edits"' in reward_manager
    assert '"graph_rpo_counterfactual_delta_abs_sum"' in reward_manager
    assert 'tag_rate = sum(tagged.get(uid, 0.0) for uid in probes) / probe_total' in reward_manager
    assert 'positive_rate = sum(positive.get(uid, 0.0) for uid in probes) / probe_total' in reward_manager
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
    assert 'TRAINER_RESUME_MODE=${TRAINER_RESUME_MODE:-auto}' in paperfaithful_launcher
    assert 'RESUME_ARGS=(trainer.resume_mode="$TRAINER_RESUME_MODE")' in paperfaithful_launcher
    assert "old_policy_counterfactual_qa" in launcher
    assert "old_policy_answer_likelihood" in paperfaithful_launcher
    assert "GRAPH_RPO_EVALUATOR_URL:?" not in launcher
    assert "old_policy_counterfactual_qa" in smoke_launcher
    assert "serve_graph_evaluator_smoke.py" not in smoke_launcher
    assert "global_step_174" in smoke_launcher
    assert "python -m verl.model_merger merge --backend fsdp" in smoke_launcher
    assert 'find -L "$MODEL_PATH"' in smoke_launcher
    assert "EXPECTED_NUM_NODES=${EXPECTED_NUM_NODES:-${#NODELIST[@]}}" in smoke_launcher
    assert "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$TRAINER_NODES}" in smoke_launcher
    assert "TOTAL_TRAINING_STEPS:-1" in smoke_launcher
    assert "ROLLOUT_N:-2" in smoke_launcher
    assert "VAL_BEFORE_TRAIN:-False" in smoke_launcher
    assert "SAVE_ROLLOUT_DATA:-1" in smoke_launcher
    assert "b968826d9c46dd6066d109eabc6255188de91218" in audit_smoke_launcher
    assert "#SBATCH -N 4" in audit_smoke_launcher
    assert "export EXPECTED_NUM_NODES=4" in audit_smoke_launcher
    assert "export TRAIN_BATCH_SIZE=6" in audit_smoke_launcher
    assert "export ROLLOUT_N=8" in audit_smoke_launcher
    assert "export PPO_MINI_BATCH_SIZE=3" in audit_smoke_launcher
    assert "export LORA_RANK=32" in audit_smoke_launcher
    assert "export LORA_ALPHA=32" in audit_smoke_launcher
    assert "export GRAPH_RPO_CREDIT_BACKEND=reference_answer_likelihood" in audit_smoke_launcher
    assert "unset RESUME_CHECKPOINT_PATH RESUME_CHECKPOINT_ROOT" in audit_smoke_launcher
    assert "adapter_model.safetensors" in audit_smoke_launcher
    assert "reference_creditable_edits:[1-9]" in audit_smoke_launcher
    assert "reference_scored_states:[1-9]" in audit_smoke_launcher
    assert "reference_delta_abs_sum:" in audit_smoke_launcher
    assert "audit_bc_judge_results.py" in audit_smoke_launcher
    assert "GRAPH_RPO_DELTA_SCALE=${GRAPH_RPO_DELTA_SCALE:-1.0}" in audit_smoke_launcher
    assert "audit_counterfactual_graph_credit.py" in audit_smoke_launcher
    assert '--expected-delta-scale "$GRAPH_RPO_DELTA_SCALE"' in audit_smoke_launcher
    assert "--fail-on-semantic-noop" in audit_smoke_launcher
    assert "SMOKE + JUDGE AUDIT COMPLETED" in audit_smoke_launcher
    assert 'export EXPERIMENT_NAME="train_ctxgraph_bc_8b_${RUN_TAG}_${RUN_TS}"' in zeroshot_pilot_launcher
    assert 'export CHECKPOINT_ROOT="$SCRATCH_ROOT/context-graph-ckpts/$EXPERIMENT_NAME"' in zeroshot_pilot_launcher
    assert "export TRAIN_BATCH_SIZE=6" in zeroshot_pilot_launcher
    assert "export ROLLOUT_N=8" in zeroshot_pilot_launcher
    assert "export PPO_MINI_BATCH_SIZE=3" in zeroshot_pilot_launcher
    assert "export LORA_RANK=32" in zeroshot_pilot_launcher
    assert "export CONTEXT_LENGTH=32768" in zeroshot_pilot_launcher
    assert "export TRAINER_RESUME_MODE=disable" in zeroshot_pilot_launcher
    assert "isolated GraphRPO output path already exists" in zeroshot_pilot_launcher
    assert "isolated zero-shot GraphRPO run unexpectedly resumed a checkpoint" in zeroshot_pilot_launcher
    assert "adapter_model.safetensors" in zeroshot_pilot_launcher
    assert "old_policy_answer_likelihood" in old_policy_smoke_launcher
    assert "TOTAL_TRAINING_STEPS=2" in old_policy_smoke_launcher
    assert "GRAPH_RPO_ALPHA=0.1" in old_policy_smoke_launcher
    assert "GRAPH_RPO_DELTA_MAX=0.25" in old_policy_smoke_launcher
    assert "BC_REQUIRE_WANDB=1" in old_policy_smoke_launcher
    assert "old_policy_counterfactual_qa" in counterfactual_smoke_launcher
    assert "GRAPH_RPO_COUNTERFACTUAL_SAMPLES" in counterfactual_smoke_launcher
    assert "GRAPH_RPO_COUNTERFACTUAL_ENABLE_THINKING=False" in counterfactual_smoke_launcher
    assert "RUN_TAG=${RUN_TAG:-graphrpo_counterfactual_4n_bs3_n8_2step}" in counterfactual_smoke_launcher
    assert "audit_counterfactual_graph_credit.py" in counterfactual_smoke_launcher
    assert "graph_rpo_counterfactual_probe_rollouts" in counterfactual_smoke_launcher
    assert "TOTAL_TRAINING_STEPS=2" in counterfactual_smoke_launcher
    assert "--dependency=afterok:" in counterfactual_stage2_submitter
    assert 'if [ -n "${SLURM_JOB_ID:-}" ]' in counterfactual_stage2_submitter
    assert "did not return a valid smoke job ID" in counterfactual_stage2_submitter
    assert "returned no valid formal job ID" in counterfactual_stage2_submitter
    assert "smoke_train_bc_ctxgraph_8b_graphrpo_counterfactual_4node_2step.sh" in counterfactual_stage2_submitter
    assert "train_bc_ctxgraph_8b_graphrpo_5node_48h.sh" in counterfactual_stage2_submitter
    assert "TRAINER_NODES=$((EXPECTED_NUM_NODES - 1))" in launcher
    assert "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((TRAINER_NODES * 8))}" in launcher

    base_launcher = (
        root / "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh"
    ).read_text(encoding="utf-8")
    assert '$NUM_NODES nodes [1 search + $((NUM_NODES - 1)) trainer]' in base_launcher
    assert 'trainer.rollout_data_dir=$ROLLOUT_DATA_DIR' in base_launcher
    assert "graph_rpo_delta_scale=$GRAPH_RPO_DELTA_SCALE" in base_launcher
    assert "GRAPH_RPO_DELTA_SCALE=${GRAPH_RPO_DELTA_SCALE:-1.0}" in launcher
    assert "GRAPH_RPO_DELTA_SCALE=${GRAPH_RPO_DELTA_SCALE:-1.0}" in zeroshot_pilot_launcher
    assert "audit_counterfactual_graph_credit.py" in zeroshot_pilot_launcher
    assert "GRAPH_RPO_AUDIT_MAX_CLIP_RATE" in zeroshot_pilot_launcher
    assert "raw_abs_delta_distribution" in matched_scale_submitter
    assert "float(stats['p80']) / float(sys.argv[2])" in matched_scale_submitter
    assert "GRAPH_RPO_ALPHA=0.0" in matched_scale_submitter
    assert "GRAPH_RPO_ALPHA=0.1" in matched_scale_submitter
    assert matched_scale_submitter.count("sbatch --parsable") == 2
    assert "raw_abs_delta_distribution" in scale_smoke_submitter
    assert "GRAPH_RPO_DELTA_SCALE=$GRAPH_RPO_DELTA_SCALE" in scale_smoke_submitter
    assert '${SLURM_JOB_ID:-}' in scale_smoke_submitter
    assert 'bash "$SMOKE_SCRIPT"' in scale_smoke_submitter
    assert "direct GraphRPO smoke requires a four-node allocation" in scale_smoke_submitter
    assert scale_smoke_submitter.count("sbatch --parsable") == 1
