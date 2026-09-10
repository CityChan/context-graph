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
    assert "unique_env_stats" in source
    assert '"reward/correctness"' in source
    assert '"reward/task"' in source
    assert '"reward/correctness_reward"' in source
    assert '"reward/soft_format_valid"' in source
    assert '"reward/soft_format_reward"' in source
    assert '"graphrpo/controller_valid_edits"' in source
    assert '"graphrpo/credited_edits"' in source
    assert '"rewards/correctness_reward_func/mean"' not in source
    assert '"rewards/soft_format_reward_func/mean"' not in source
    assert '"rollout_per_token_logps"' in source
    assert "old_per_token_logps = per_token_logps.detach()" not in source
    assert "if old_per_token_logps is None" in source
    assert '"foldgrpo_loss_weights"' in source
    assert "1.0 / len(outputs)" in source


def test_training_entrypoint_does_not_import_qerl_reward_stack():
    source = (ROOT / "trl_agent" / "train.py").read_text(encoding="utf-8")
    modeling = (ROOT / "trl_agent" / "modeling.py").read_text(encoding="utf-8")
    assert "from qerl import" not in source
    assert "from trl_agent.modeling import build_model_and_peft" in source
    assert "from utils.rewards import" not in modeling
    assert "prepare_model_for_kbit_training" in modeling


def test_graphtrl_setup_imports_real_training_entrypoint():
    source = (ROOT / "scripts" / "setup_graphtrl_env.sh").read_text(encoding="utf-8")
    assert '"ray[default]"' in source
    assert '"codetiming"' in source
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
    assert "export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}" in launcher
    assert "export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-128}" in launcher
    assert "train_gsm8k_trl_agent_lora32.sh" in launcher


@pytest.mark.parametrize("kind", ["foldagent", "contextgraph"])
def test_formal_launchers_use_full_dataset_and_periodic_checkpoints(kind):
    launcher = (
        ROOT / "scripts" / f"train_gsm8k_trl_{kind}_lora32_200step.sh"
    ).read_text(encoding="utf-8")
    assert f"export AGENT_KIND={kind}" in launcher
    assert "export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}" in launcher
    assert "export SAVE_STEPS=${SAVE_STEPS:-50}" in launcher
    assert "export TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-0}" in launcher
    assert "train_gsm8k_trl_agent_lora32.sh" in launcher


def test_shared_launcher_preflights_the_selected_gpu_and_defaults_to_full_data():
    source = (
        ROOT / "scripts" / "train_gsm8k_trl_agent_lora32.sh"
    ).read_text(encoding="utf-8")
    visibility = "export CUDA_VISIBLE_DEVICES=${TRL_AGENT_CUDA_DEVICE:-0}"
    preflight = 'python -c "import ctypes, omegaconf'
    assert "TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-0}" in source
    assert "TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-}" in source
    assert 'TRAIN_DATA_ARGS+=(--train-data-path "$TRAIN_DATA_PATH")' in source
    assert visibility in source
    assert source.index(visibility) < source.index(preflight)
    assert "CUDA_VISIBLE_DEVICES=0 accelerate launch" not in source
    assert "'reward/correctness'" in source
    assert "'reward/soft_format_valid'" in source
    assert "'training/old_policy_logps_recomputed'" in source
    assert "rows[-1]['clip_ratio/region_mean'] == 0.0" in source
    assert "GraphRPO plumbing audit: OK" in source
    assert "no valid controller edit reached the graph trace" in source
    assert "no counterfactual QA probe was generated" in source
    assert "all counterfactual QA probes violated the answer contract" in source
    assert "inspect graphrpo/delta_abs_sum for utility signal" in source
    assert "no nonzero GraphRPO edit credit was assigned" not in source


def test_contextgraph_recipe_uses_deterministic_credited_controller_probes():
    source = (
        ROOT / "recipes" / "trl_agent" / "contextgraph_gsm8k.yaml"
    ).read_text(encoding="utf-8")
    assert "graph_controller_temperature: 0.0" in source
    assert "graph_rpo_counterfactual_temperature: 0.0" in source
    assert "graph_rpo_operation_costs:" in source


def test_contextgraph_bc_qwen3_8b_lora_50step_launcher_is_protocol_labeled():
    source = (
        ROOT
        / "scripts"
        / "train_bc_trl_contextgraph_qwen3_8b_lora32_50step_2node.sh"
    ).read_text(encoding="utf-8")
    recipe = (
        ROOT / "recipes" / "trl_agent" / "contextgraph_browsecomp_plus.yaml"
    ).read_text(encoding="utf-8")
    assert "export AGENT_KIND=contextgraph" in source
    assert "data/bc_train.parquet" in source
    assert "MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}" in source
    assert "TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}" in source
    assert "TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-128}" in source
    assert "contextgraph-bc-qwen3-8b-lora32-50step" in source
    assert "contextgraph_browsecomp_plus.yaml" in source
    assert "envs/search_server.py" in source
    assert 'if [ "${#BC_TRL_NODES[@]}" -lt 2 ]; then' in source
    assert "extra allocation node(s) idle" in source
    assert "workflow: search_graph" in recipe
    assert "must_search: true" in recipe
    assert "graph_rpo_credit_backend: old_policy_counterfactual_qa" in recipe
    assert "merge: 0.0" in recipe
