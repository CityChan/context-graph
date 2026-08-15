import numpy as np
import pytest


def test_grpo_outcome_advantage_groups_by_prompt_uid():
    torch = pytest.importorskip("torch")
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    rewards = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    response_mask = torch.ones_like(rewards)
    prompt_uids = np.array(["a", "a", "b", "b"], dtype=object)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=prompt_uids,
    )

    expected = torch.tensor(
        [
            [-0.7071058, -0.7071058],
            [0.7071058, 0.7071058],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    torch.testing.assert_close(advantages, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(returns, expected, rtol=1e-5, atol=1e-5)
