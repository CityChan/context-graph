from types import SimpleNamespace

import numpy as np
import pytest

from agents.rollout_status import classify_rollout_status


def _terminal_rewards(values: list[float], width: int = 3):
    torch = pytest.importorskip("torch")
    rewards = torch.zeros((len(values), width), dtype=torch.float32)
    rewards[:, -1] = torch.tensor(values, dtype=torch.float32)
    return rewards


def _paper_advantages(
    terminal_rewards: list[float], process_rewards, mode: str = "paper"
):
    torch = pytest.importorskip("torch")
    from verl.trainer.ppo.core_algos import compute_foldgrpo_advantage

    batch_size, width = process_rewards.shape
    advantages, _ = compute_foldgrpo_advantage(
        token_level_rewards=_terminal_rewards(terminal_rewards, width),
        response_mask=torch.ones((batch_size, width), dtype=torch.float32),
        index=np.array(["prompt"] * batch_size, dtype=object),
        gen_uid=np.array([f"generation-{i}" for i in range(batch_size)], dtype=object),
        process_reward_mask=process_rewards,
        config={"foldgrpo_process_reward_mode": mode},
    )
    return advantages


def test_paper_process_reward_uses_clipped_terminal_plus_q():
    torch = pytest.importorskip("torch")
    q = torch.zeros((2, 3), dtype=torch.float32)
    q[1, 0] = -1.0

    advantages = _paper_advantages([0.0, 1.0], q)
    expected_magnitude = 0.5 / (torch.std(torch.tensor([0.0, 1.0])) + 1e-6)

    assert advantages[0, 0].item() == pytest.approx(-expected_magnitude.item())
    assert advantages[1, 0].item() == pytest.approx(-expected_magnitude.item())
    assert advantages[1, 1].item() == pytest.approx(expected_magnitude.item())


def test_paper_process_reward_remains_finite_in_degenerate_groups():
    torch = pytest.importorskip("torch")
    q = torch.zeros((2, 3), dtype=torch.float32)
    q[0, 0] = -0.2

    advantages = _paper_advantages([1.0, 1.0], q)

    assert torch.isfinite(advantages).all()
    assert advantages[0, 0].item() == pytest.approx(-0.2)
    assert advantages[0, 1].item() == pytest.approx(0.0)
    assert advantages[1].abs().sum().item() == pytest.approx(0.0)


def test_signed_process_reward_survives_on_failed_degenerate_group():
    torch = pytest.importorskip("torch")
    q = torch.zeros((2, 3), dtype=torch.float32)
    q[0, 1] = -1.0

    advantages = _paper_advantages([0.0, 0.0], q, mode="paper_signed")

    assert torch.isfinite(advantages).all()
    assert advantages[0, 1].item() == pytest.approx(-1.0)
    assert advantages[0, 0].item() == pytest.approx(0.0)
    assert advantages[1].abs().sum().item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "loss_agg_mode", ["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"]
)
def test_aggregate_loss_is_zero_for_fully_masked_microbatch(loss_agg_mode):
    torch = pytest.importorskip("torch")
    from verl.trainer.ppo.core_algos import agg_loss

    loss = agg_loss(
        loss_mat=torch.full((1, 3), torch.nan, dtype=torch.float32, requires_grad=True),
        loss_mask=torch.zeros((1, 3), dtype=torch.float32),
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=0,
        global_batch_size=0,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_rollout_status_separates_length_turn_timeout_and_finish():
    token_limit = classify_rollout_status(
        response_tokens=99,
        response_limit=100,
        main_context_tokens=99,
        working_context_limit=100,
        is_finish=False,
        iteration=12,
        max_turn=100,
        timed_out=False,
    )
    assert token_limit["overlong"] == 1
    assert token_limit["hit_token_limit"] == 1
    assert token_limit["hit_max_turn"] == 0
    assert token_limit["termination_reason"] == "token_limit"

    max_turn = classify_rollout_status(
        response_tokens=40,
        response_limit=100,
        main_context_tokens=40,
        working_context_limit=100,
        is_finish=False,
        iteration=100,
        max_turn=100,
        timed_out=False,
    )
    assert max_turn["overlong"] == 0
    assert max_turn["hit_max_turn"] == 1
    assert max_turn["termination_reason"] == "max_turn"

    finished = classify_rollout_status(
        response_tokens=100,
        response_limit=100,
        main_context_tokens=100,
        working_context_limit=100,
        is_finish=True,
        iteration=100,
        max_turn=100,
        timed_out=True,
    )
    assert finished["overlong"] == 0
    assert finished["no_finish"] == 0
    assert finished["hit_timeout"] == 0
    assert finished["termination_reason"] == "finish"


def test_reward_metrics_do_not_treat_optimization_mask_as_overlong():
    pytest.importorskip("torch")
    from verl.workers.reward_manager.agent import AgentLoopRewardManager

    data = SimpleNamespace(
        non_tensor_batch={
            "gen_uid": np.array(["a", "b"], dtype=object),
            "mask_rollout": np.array([True, True], dtype=object),
            "overlong": np.array([False, True], dtype=object),
            "no_finish": np.array([False, True], dtype=object),
            "is_finish": np.array([True, False], dtype=object),
            "__num_turns__": np.array([2, 3], dtype=np.int32),
        },
        meta_info={},
        __len__=lambda self: 2,
    )
    # Special methods are looked up on the type, not the instance.
    data = type(
        "Batch",
        (),
        {
            "non_tensor_batch": data.non_tensor_batch,
            "meta_info": {},
            "__len__": lambda self: 2,
        },
    )()

    metrics = AgentLoopRewardManager.__new__(AgentLoopRewardManager)._compute_batch_metrics(
        data, [0.0, 1.0]
    )

    assert metrics["overlong_rate"][0] == pytest.approx(0.5)
    assert metrics["no_finish_rate"][0] == pytest.approx(0.5)
    assert metrics["finish_rate"][0] == pytest.approx(0.5)
