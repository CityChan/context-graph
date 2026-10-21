import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agents.scienceworld_training import NoToolGuard, training_reward
from envs.scienceworld_env import ScienceWorldEnv


@pytest.mark.parametrize("completed,peak,natural,token,no_tool,failed,expected", [
    (True, 100, True, True, True, False, 1.),
    (False, 60, True, False, False, False, .09),
    (False, -1, True, False, False, False, 0.),
    (False, 120, True, False, False, False, .15),
    (False, 60, False, True, False, False, -.1),
    (False, 60, False, False, True, False, -.1),
    (False, 60, False, False, False, False, 0.),
    (True, 100, True, False, False, True, 0.),
])
def test_training_reward_precedence(completed, peak, natural, token, no_tool, failed, expected):
    assert training_reward(completed=completed, peak_score=peak, natural_end=natural,
                           token_limit=token, no_tool_limit=no_tool, env_failed=failed) == pytest.approx(expected)


def test_nonfinite_progress_is_not_silently_repaired():
    with pytest.raises(ValueError, match="finite"):
        training_reward(completed=False, peak_score=float("nan"), natural_end=True,
                        token_limit=False, no_tool_limit=False)


def test_no_tool_guard_resets_on_calls_and_stops_at_five():
    guard = NoToolGuard(5)
    assert all(guard.observe(False) for _ in range(4))
    assert guard.observe(True)
    assert all(guard.observe(False) for _ in range(4))
    assert not guard.observe(False)
    assert not guard.observe(True)  # cap cannot be undone by a later call
    disabled = NoToolGuard()
    assert all(disabled.observe(False) for _ in range(10))


class Simulator:
    sequence = [(60, False), (-1, True)]

    def __init__(self, **kwargs):
        self.steps = iter(self.sequence)

    def load(self, **kwargs):
        assert kwargs["generateGoldPath"] is False

    def reset(self):
        return "room", {"score": 0}

    def get_task_description(self):
        return "Boil water."

    def step(self, command):
        score, done = next(self.steps)
        return "observed", score, done, {"score": score}

    def close(self):
        pass


def item():
    return SimpleNamespace(non_tensor_batch={
        "ability": np.array(["ScienceWorld@real"]), "uid": "q", "gen_uid": "episode",
        "extra_info": np.array([{"task_name": "boil", "variation_idx": 0,
                                 "workflow": "scienceworld_graph"}], dtype=object),
    })


ACTION = "<function=action><parameter=command>look around</parameter></function>"


@pytest.mark.parametrize("step_limit,natural", [(10, 1), (2, 0)])
def test_adapter_tracks_peak_and_distinguishes_step_cap(monkeypatch, step_limit, natural):
    class PositiveEnd(Simulator):
        sequence = [(60, False), (40, True)]
    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=PositiveEnd))
    config = SimpleNamespace(plugin=SimpleNamespace(scienceworld_max_steps=step_limit))
    env = ScienceWorldEnv(config, None, "ScienceWorld@real")
    async def run():
        await env.init_env(item())
        await env.run_action(ACTION)
        await env.run_action(ACTION)
        assert env.stats["environment_peak_score"] == 60
        assert env.stats["environment_score"] == 40
        assert env.stats["environment_natural_end"] == natural
        # The native environment API remains binary even with partial progress.
        assert (await env.get_reward(item(), [], SimpleNamespace(is_train=True)))[1] == 0
        await env.init_env(item())
        assert env.stats["environment_peak_score"] == 0
        env.close()
    asyncio.run(run())


def setup_rollout(monkeypatch, responses, *, train=True, max_turn=20):
    from agents import graph_agent_isolated
    from scripts.eval_discoverybench_qwen35 import config_for
    from tests.test_session_restart import Client, Tokenizer

    monkeypatch.setitem(sys.modules, "scienceworld", SimpleNamespace(ScienceWorldEnv=Simulator))
    config = config_for("contextgraph", 32768)
    config.algorithm.adv_estimator = "graphrpo"
    config.algorithm.graphrpo_require_binary_reward = False
    plugin = config.actor_rollout_ref.rollout.plugin
    plugin.workflow = "scienceworld_graph"
    plugin.scienceworld_training_reward_profile = "v3_long"
    plugin.scienceworld_prompt_profile = "legacy"
    plugin.graph_rpo_credit_backend = "outcome_only"
    plugin.graph_rpo_scope_process_reward = False
    plugin.process_reward = ["graphrpo"]
    plugin.final_answer_reserve = 0
    plugin.must_finish = False
    plugin.must_branch = False
    plugin.consolidation_interval = 0
    plugin.initial_consolidation_turn = 0
    plugin.max_turn = plugin.val_max_turn = max_turn
    plugin.branch_len = 12000
    client = Client(responses)
    context = SimpleNamespace(config=config, is_train=train, global_step=0,
                              tokenizer=Tokenizer(), llm_client=client)
    return graph_agent_isolated, context, client


@pytest.mark.parametrize("train,expected", [(True, .09), (False, 0.)])
def test_real_loop_applies_training_reward_only_in_training(monkeypatch, train, expected):
    module, context, client = setup_rollout(monkeypatch, [ACTION, ACTION], train=train)
    outputs = asyncio.run(module.process_item(item(), context))
    assert len(client.calls) == 2
    for out in outputs:
        assert out.reward_score == pytest.approx(expected)
        assert out.extra_fields["env_stats"]["task_reward"] == 0
        assert out.extra_fields["native_task_reward"] == 0
        assert out.extra_fields["training_reward_applied"] == train
        assert out.extra_fields["no_finish"] == 0
        assert out.extra_fields["env_stats"]["environment_peak_score"] == 60
        assert not any(out.extra_fields["graph_edit_credit_mask"])
        assert out.extra_fields["graph_rpo_credit_backend"] == "outcome_only"


@pytest.mark.parametrize("branch", [False, True])
def test_no_tool_cap_in_main_or_branch_preserves_negative_training_signal(monkeypatch, branch):
    responses = ["No tool here."] * 5
    if branch:
        responses.insert(0, "<function=branch><parameter=description>inspect</parameter><parameter=prompt>Inspect the room</parameter></function>")
    module, context, client = setup_rollout(monkeypatch, responses)
    outputs = asyncio.run(module.process_item(item(), context))
    assert len(client.calls) == 5 + int(branch)
    assert len(outputs) == 1 + int(branch)
    assert all(out.reward_score == -.1 for out in outputs)
    assert all(out.extra_fields["termination_reason"] == "no_tool_limit" for out in outputs)
    assert all(not out.extra_fields["mask_rollout"] for out in outputs)
    assert any(-1 in out.extra_fields["process_reward_mask"] for out in outputs)


def test_turn_truncation_drops_partial_progress_without_limit_penalty(monkeypatch):
    module, context, _ = setup_rollout(monkeypatch, [ACTION], max_turn=1)
    outputs = asyncio.run(module.process_item(item(), context))
    assert outputs[0].reward_score == 0
    assert outputs[0].extra_fields["env_stats"]["environment_peak_score"] == 60
    assert outputs[0].extra_fields["termination_reason"] == "max_turn"


def test_outcome_only_controller_uses_no_scoring_calls(monkeypatch):
    import json
    class LongerSimulator(Simulator):
        sequence = [(30, False), (60, False), (100, True)]
    decision = json.dumps(dict(action='prune', candidate_indices=[0], summary='', relation='semantic'))
    module, context, client = setup_rollout(monkeypatch, [ACTION, ACTION, decision, ACTION])
    monkeypatch.setitem(sys.modules, 'scienceworld', SimpleNamespace(ScienceWorldEnv=LongerSimulator))
    plugin = context.config.actor_rollout_ref.rollout.plugin
    plugin.consolidation_interval = 2
    async def no_probe(**kwargs):
        pytest.fail('outcome_only must not invoke a graph evaluator or counterfactual probe')
    monkeypatch.setattr(module, 'assign_graph_edit_credits', no_probe)
    outputs = asyncio.run(module.process_item(item(), context))
    assert len(client.calls) == 4
    assert outputs[0].reward_score == 1
    assert any(outputs[0].extra_fields['graph_decision_mask'])
    assert not any(outputs[0].extra_fields['graph_edit_credit_mask'])
    assert outputs[0].extra_fields['env_stats']['graph_rpo_outcome_only'] == 1


def test_loop_token_exhaustion_keeps_negative_reward(monkeypatch):
    module, context, client = setup_rollout(monkeypatch, [ACTION])
    original_completion = client.create_completion
    async def bounded_completion(ids, **kwargs):
        # Match CallLLM's real refusal when fewer than ten tokens remain.
        if kwargs['max_len'] - len(ids) < 10:
            return None
        return await original_completion(ids, **kwargs)
    monkeypatch.setattr(client, 'create_completion', bounded_completion)
    original = module.step_preserving_final_answer
    async def exhaust_budget(agent, *args, **kwargs):
        response = await original(agent, *args, **kwargs)
        # Exhaust the budget after a real action response, before its observation.
        agent.config.response_length = agent.response_tokens_used() + 5
        agent.working_context_limit = len(agent.context()) + 5
        return response
    monkeypatch.setattr(module, 'step_preserving_final_answer', exhaust_budget)
    outputs = asyncio.run(module.process_item(item(), context))
    assert outputs[0].reward_score == -.1
    assert outputs[0].extra_fields['hit_token_limit'] == 1
    assert outputs[0].extra_fields['env_stats']['environment_peak_score'] == 60
    assert not outputs[0].extra_fields['mask_rollout']


def compose_profile():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path("verl/trainer/config").resolve()), version_base=None):
        return compose(config_name="scienceworld_graphrpo_v3_long")


def test_profile_resolves_and_rejects_incompatible_overrides(tmp_path):
    from scripts.train_scienceworld_graphrpo import validate_config, write_manifest
    from verl.utils.config import validate_config as validate_verl_config
    from omegaconf import OmegaConf
    config = compose_profile()
    validate_config(config)
    assert config.data.train_batch_size == 8 and config.actor_rollout_ref.rollout.n == 4
    assert config.algorithm.graphrpo_dynamic_max_gen_batches == 3
    config.actor_rollout_ref.model.path = "test-model"
    validate_verl_config(config, use_reference_policy=True, use_critic=False)
    config.trainer.default_local_dir = str(tmp_path)
    config.scienceworld_profile_prepare_only = True
    write_manifest(config)
    saved = OmegaConf.load(next(tmp_path.glob("protocol-*.yaml")))
    assert saved.actor_rollout_ref.rollout.response_length == 57344
    assert saved.algorithm.graphrpo_norm_adv_by_std is False
    assert saved.actor_rollout_ref.rollout.plugin.graph_action_protocol == 'explicit_tools'
    assert saved.actor_rollout_ref.rollout.plugin.structured_graph_controller is False
    config.algorithm.mask_overlong = True
    with pytest.raises(ValueError, match="exhausted"):
        validate_config(config)


def test_profile_dynamic_sampling_keeps_process_ties_and_caps_extra_batches():
    from tests.test_training_hardening import batch
    from verl.trainer.ppo.dynamic_groups import informative_groups, sample_groups
    config = compose_profile().algorithm
    tied = batch((0., 0., 0.))
    assert informative_groups(tied, config)[0] == []
    tied.batch['process_reward_mask'][1, 0] = -1
    assert informative_groups(tied, config)[0] == [[0, 1, 2]]
    generated = []
    def generate(value):
        generated.append(value)
        return batch((-.1, -.1, -.1), uid=str(value))
    remaining, metrics = iter([1, 2, 3]), {}
    result = sample_groups(0, remaining, generate, config, 8, metrics)
    assert generated == [0, 1, 2] and next(remaining) == 3
    assert metrics['graphrpo/dynamic_min_groups_met'] == 0
    assert result.meta_info['dynamic_empty'] is True


def test_native_validation_and_training_use_same_prompt(monkeypatch):
    seen = []
    from agents import graph_agent_isolated
    original = graph_agent_isolated.create_chat
    def record(*args, **kwargs):
        messages = original(*args, **kwargs)
        seen.append((kwargs['scienceworld_prompt_profile'], messages))
        return messages
    monkeypatch.setattr(graph_agent_isolated, 'create_chat', record)
    for train in (True, False):
        module, context, _ = setup_rollout(monkeypatch, [ACTION, ACTION], train=train)
        asyncio.run(module.process_item(item(), context))
    assert seen[0] == seen[1]
    assert seen[0][0] == 'legacy'
