from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agents.prompts import create_chat
from trl_agent.rollout import ColocatedVLLMBatcher, RolloutResult, chosen_token_logprobs


ROOT = Path(__file__).resolve().parents[1]


class _Logprob:
    def __init__(self, value: float):
        self.logprob = value


def test_chosen_token_logprobs_normalizes_vllm_mapping():
    assert chosen_token_logprobs(
        [11, 22], [{11: _Logprob(-0.1)}, {22: _Logprob(-0.2)}]
    ) == pytest.approx([-0.1, -0.2])


def test_chosen_token_logprobs_rejects_missing_chosen_token():
    with pytest.raises(RuntimeError, match="absent"):
        chosen_token_logprobs([11], [{22: _Logprob(-0.2)}])


def test_async_batcher_coalesces_ready_agent_turns():
    class FakeBatcher(ColocatedVLLMBatcher):
        def __init__(self):
            super().__init__(object(), batch_wait_ms=1)
            self.batch_sizes = []

        def _run_batch(self, requests):
            self.batch_sizes.append(len(requests))
            return [
                RolloutResult(token_ids=request.prompt_ids[-1:], log_probs=[-0.5])
                for request in requests
            ]

    async def run():
        batcher = FakeBatcher()
        results = await asyncio.gather(
            batcher.generate(
                request_id="a", prompt_ids=[1, 2], sampling_params={}, image_data=None
            ),
            batcher.generate(
                request_id="b", prompt_ids=[3, 4], sampling_params={}, image_data=None
            ),
        )
        return batcher, results

    batcher, results = asyncio.run(run())
    assert batcher.batch_sizes == [2]
    assert [result.token_ids for result in results] == [[2], [4]]


def test_math_branch_prompt_exposes_branch_without_graph_operations():
    messages = create_chat("What is 2 + 2?", workflow="math_branch")
    system = messages[0]["content"]
    assert "BEGIN FUNCTION #3: branch" in system
    assert ": merge ----" not in system
    assert ": add_edge ----" not in system


def test_trainer_keeps_attention_and_policy_masks_separate():
    source = (ROOT / "trl_agent" / "trainer.py").read_text(encoding="utf-8")
    assert '"completion_attention_mask"' in source
    assert '"completion_mask"' in source
    assert "compute_foldgrpo_advantage(" in source
    assert "compute_graphrpo_advantage(" in source
    assert "compute_graphrpo_loss_weights(" in source
    assert "episode_ids" in source
    assert "torch.tensor_split" in source


def test_training_entrypoint_does_not_import_qerl_reward_stack():
    source = (ROOT / "trl_agent" / "train.py").read_text(encoding="utf-8")
    modeling = (ROOT / "trl_agent" / "modeling.py").read_text(encoding="utf-8")
    assert "from qerl import" not in source
    assert "from trl_agent.modeling import build_model_and_peft" in source
    assert "from utils.rewards import" not in modeling
    assert "prepare_model_for_kbit_training" in modeling


def test_graphtrl_setup_imports_real_training_entrypoint():
    source = (ROOT / "scripts" / "setup_graphtrl_env.sh").read_text(encoding="utf-8")
    assert "import tensordict, torch, trl, trl_agent.train, vllm" in source
    assert "OmegaConf.load('recipes/trl_agent/foldagent_gsm8k.yaml')" in source


@pytest.mark.parametrize(
    ("name", "kind", "estimator"),
    [
        ("foldagent_gsm8k.yaml", "foldagent", "foldgrpo"),
        ("contextgraph_gsm8k.yaml", "contextgraph", "graphrpo"),
    ],
)
def test_launch_configs_and_wrappers_are_wired(name, kind, estimator):
    config = (ROOT / "recipes" / "trl_agent" / name).read_text(encoding="utf-8")
    launcher = (ROOT / "scripts" / f"train_gsm8k_trl_{kind}_lora32_10step.sh").read_text(
        encoding="utf-8"
    )
    assert f"adv_estimator: {estimator}" in config
    assert f"export AGENT_KIND={kind}" in launcher
    assert "train_gsm8k_trl_agent_lora32.sh" in launcher
