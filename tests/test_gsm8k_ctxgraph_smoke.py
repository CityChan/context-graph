from pathlib import Path


SCRIPT = Path("scripts/smoke_train_gsm8k_ctxgraph_graphrpo_1node_10step.sh")
QERL_ALIGNED_SCRIPT = Path(
    "scripts/smoke_train_gsm8k_ctxgraph_graphrpo_nocredit_lora32_10step.sh"
)


def test_ctxgraph_smoke_exercises_real_contextgraph_graphrpo_path():
    source = SCRIPT.read_text()

    required = [
        "python -m scripts.train_graph",
        "algorithm.adv_estimator=graphrpo",
        "actor_rollout_ref.actor.policy_loss.loss_mode=graphrpo",
        "default_agent_loop=context_graph_isolated_agent",
        "+actor_rollout_ref.rollout.plugin.workflow=math_graph",
        "+actor_rollout_ref.rollout.plugin.structured_graph_controller=True",
        "+actor_rollout_ref.rollout.plugin.must_branch=True",
        "+actor_rollout_ref.rollout.plugin.graph_rpo_credit_backend=old_policy_answer_likelihood",
        "+actor_rollout_ref.rollout.plugin.graph_rpo_scope_process_reward=False",
        r"\[GRAPH CONTROLLER (MERGE|PRUNE|ADD_EDGE|SELECT)\]",
        "graphrpo/old_policy_scored_states",
        "actor/grad_norm:",
    ]
    for setting in required:
        assert setting in source


def test_ctxgraph_smoke_is_self_contained_and_uses_node_local_caches():
    source = SCRIPT.read_text()

    assert "unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE RAY_ADDRESS OPENAI_API_KEY OPENAI_URL LOCAL_SEARCH_URL" in source
    assert "TORCHDYNAMO_DISABLE=1" in source
    assert "actor_rollout_ref.rollout.enforce_eager=True" in source
    assert "libnvrtc.so.12" in source
    assert "gsm8k-contextgraph" in source


def test_ctxgraph_runtime_enforces_branch_and_can_disable_external_scope_judge():
    agent_source = Path("agents/graph_agent_isolated.py").read_text(encoding="utf-8")

    assert "and must_branch" in agent_source
    assert "and not branches" in agent_source
    assert "and (not must_branch or bool(branches))" in agent_source
    assert "fn_call['function'] == 'think'" in agent_source
    assert 'required_labels = ["graphrpo"]' in agent_source
    assert 'getattr(config.plugin, "graph_rpo_scope_process_reward", True)' in agent_source


def test_qerl_aligned_smoke_has_zero_local_credit_lora_and_matched_reward_scale():
    source = QERL_ALIGNED_SCRIPT.read_text()

    required = [
        "algorithm.graphrpo_alpha=0.0",
        "algorithm.graphrpo_beta=0.0",
        "algorithm.graphrpo_require_binary_reward=False",
        "actor_rollout_ref.model.lora_rank=32",
        "actor_rollout_ref.model.lora_alpha=32",
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.actor.optim.lr=1e-5",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "+actor_rollout_ref.rollout.plugin.math_correctness_reward_weight=2.0",
        "+actor_rollout_ref.rollout.plugin.math_format_reward_weight=0.2",
        'trainer.logger=\'["console","wandb"]\'',
        "actor/lora_adapter",
    ]
    for setting in required:
        assert setting in source
