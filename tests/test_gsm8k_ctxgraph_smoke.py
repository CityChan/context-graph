from pathlib import Path


SCRIPT = Path("scripts/smoke_train_gsm8k_ctxgraph_graphrpo_1node_10step.sh")


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
