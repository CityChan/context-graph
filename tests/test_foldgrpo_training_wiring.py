from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _paper_reward_section(relative_path: str) -> str:
    source = _read(relative_path)
    start = source.index("paper process rewards")
    end = source.index("if score[1] <= 0:", start)
    return source[start:end]


def test_fractional_process_rewards_are_not_truncated():
    source = _read("verl/experimental/agent_loop/agent_loop.py")

    assert "dtype=torch.float32" in source
    assert "process_reward_mask = prm * response_mask.to(torch.float32)" in source
    assert "process_reward_mask.to(torch.float32)" in source


def test_foldagent_uses_paper_process_rewards_on_all_outcomes():
    section = _paper_reward_section("agents/fold_agent.py")

    assert "if score[1] > 0" not in section
    assert "set_process_reward(bad_turn, -1)" in section
    assert "set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)" in section
    assert "elif is_focus > 0" not in section
    assert "set_cache('reward'" not in section
    assert "for name in agent:" in section
    assert "'<function=finish>' not in str(turn)" not in section


def test_foldagent_training_selects_paper_advantage_formula():
    source = _read("scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh")
    assert "algorithm.foldgrpo_process_reward_mode=paper_signed \\" in source

    trainer = _read("verl/trainer/ppo/ray_trainer.py")
    assert "config=config" in trainer

    reward_manager = _read("verl/workers/reward_manager/agent.py")
    assert 'data.non_tensor_batch.get("mask_rollout"' not in reward_manager

    trainer = _read("verl/trainer/ppo/ray_trainer.py")
    assert "'overlong_masked'" not in trainer
    assert "'optimization_masked_rollouts'" in trainer


def test_nonfinite_actor_microbatches_are_skipped_instead_of_multiplied_by_zero():
    source = _read("verl/workers/actor/dp_actor.py")

    assert "pg_loss = pg_loss * 0.0" not in source
    assert "kl_loss = kl_loss * 0.0" not in source
    assert '"actor/skipped_nonfinite_micro_batch"' in source
    assert "did_backward" in source


def test_foldagent_penalizes_the_last_action_when_finish_is_missing():
    section = _paper_reward_section("agents/fold_agent.py")

    assert "if rollout_status['no_finish']:" in section
    assert "agent['main'].chat_completions" in section
    assert "agent['main'].set_process_reward(last_completion, -1)" in section


def test_contextgraph_inherits_the_same_paper_base_signal():
    section = _paper_reward_section("agents/graph_agent_isolated.py")

    assert "if score[1] > 0" not in section
    assert "set_process_reward(bad_turn, -1)" in section
    assert "set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)" in section
    assert "elif is_focus > 0" not in section
    assert "set_cache('reward'" not in section
    assert "for name in agent:" in section


def test_contextgraph_receives_fresh_graph_state_and_actionable_invalid_penalties():
    source = _read("agents/graph_agent_isolated.py")

    assert '"[Latest ContextGraph state]\\n"' in source
    assert "graph_turn_idx = len(agent['main'].chat) - 1" in source
    assert "graph_turn_idx, graph_invalid_penalty" in source
    assert "if rollout_status['no_finish']:" in source
    assert "agent['main'].set_process_reward(last_completion, -1)" in source


def test_contextgraph_uses_one_terminal_reward_per_episode():
    source = _read("agents/graph_agent_isolated.py")

    assert "if use_graph_reward:" in source
    assert "if use_graph_reward and name == 'main':" not in source


def test_contextgraph_prompt_matches_isolated_graph_capabilities():
    prompt = _read("agents/prompts.py")
    tools = _read("agents/tool_spec.py")

    assert "graph operations from BRANCH are invalid" in prompt
    assert "Branches can also use graph tools" not in prompt
    assert "'name': 'pass'" in tools
    assert "<function=pass></function>" in prompt


def test_contextgraph_training_selects_composite_reward_advantages_explicitly():
    source = _read("scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh")

    assert "algorithm.foldgrpo_process_reward_mode=relative_extrema \\" in source
    assert "+actor_rollout_ref.rollout.plugin.graph_invalid_penalty=-0.3 \\" in source


def test_production_wrappers_use_global_128_minibatch_arithmetic():
    for script in (
        "scripts/train_bc_foldagent_8b_5node_50step_32k_active.sh",
        "scripts/train_bc_ctxgraph_8b_5node_50step_32k_active.sh",
        "scripts/train_bc_foldagent_8b_5node_50step_64k_active.sh",
        "scripts/train_bc_ctxgraph_8b_5node_50step_64k_active.sh",
    ):
        source = _read(script)
        assert "export PPO_MINI_BATCH_SIZE=32" in source
        assert "32 per rank x 4 trainer ranks = paper-scale global 128" in source


def test_64k_training_wrappers_and_long_context_overrides_are_wired():
    wrappers = (
        "scripts/train_bc_foldagent_8b_5node_50step_64k_active.sh",
        "scripts/train_bc_ctxgraph_8b_5node_50step_64k_active.sh",
    )
    for script in wrappers:
        source = _read(script)
        assert "export PROMPT_LENGTH=8192" in source
        assert "export RESPONSE_LENGTH=57344" in source
        assert "export CONTEXT_LENGTH=65536" in source
        assert "export BC_YARN_FACTOR=2.0" in source
        assert "export BC_YARN_ORIGINAL_LENGTH=32768" in source

    for script in (
        "scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
    ):
        source = _read(script)
        assert 'if [ "$CONTEXT_LENGTH" -gt 40960 ]; then' in source
        assert "+actor_rollout_ref.model.override_config=" in source
        assert "+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides=" in source
        assert '"${LONG_CONTEXT_ARGS[@]}"' in source

    submit = _read("scripts/submit_train_bc_fa_cg_8b_5node_50step_64k_active.sh")
    for script in wrappers:
        assert script in submit


def test_foldagent_entrypoint_and_one_step_smokes_use_production_paths():
    fold_base = _read("scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh")
    assert "python -m scripts.train_fold" in fold_base
    assert "python -m scripts.train_graph" not in fold_base

    for script, base in (
        (
            "scripts/smoke_train_bc_foldagent_8b_5node_1step_32k_active.sh",
            "scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh",
        ),
        (
            "scripts/smoke_train_bc_ctxgraph_8b_5node_1step_32k_active.sh",
            "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
        ),
    ):
        source = _read(script)
        assert "export TOTAL_TRAINING_STEPS=1" in source
        assert "export TRAIN_BATCH_SIZE=4" in source
        assert "export ROLLOUT_N=2" in source
        assert "export PPO_MINI_BATCH_SIZE=2" in source
        assert f"exec bash {base}" in source


def test_five_node_idev_runner_executes_both_training_smokes():
    source = _read("scripts/smoke_train_bc_pair_8b_5node_1step_32k_active.sh")

    assert 'if [ "${#IDEV_NODES[@]}" -ne 5 ]; then' in source
    assert "smoke_train_bc_foldagent_8b_5node_1step_32k_active.sh" in source
    assert "smoke_train_bc_ctxgraph_8b_5node_1step_32k_active.sh" in source
    assert "FOLD_RC=${PIPESTATUS[0]}" in source
    assert "CTXGRAPH_RC=${PIPESTATUS[0]}" in source
    assert "idev_smoke_$(date +%Y%m%d_%H%M%S)" in source


def test_training_waits_until_search_is_actually_ready():
    for script in (
        "scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
    ):
        source = _read(script)
        assert "BC_SEARCH_TIMEOUT_SECONDS=${BC_SEARCH_TIMEOUT_SECONDS:-600}" in source
        assert 'probe "waiting for search server /search probe' in source
        assert 'for _ in $(seq 1 "$BC_SEARCH_TIMEOUT_SECONDS"); do' in source
        assert "SEARCH_OK=1" in source
        assert 'if [ "$SEARCH_OK" != "1" ]; then' in source
        assert 'kill -0 "$SEARCH_PID"' in source


def test_sab_eval_agents_emit_explicit_termination_metrics():
    for agent_path in (
        "agents/react_agent_code.py",
        "agents/fold_agent_code.py",
        "agents/graph_agent_code_isolated.py",
    ):
        source = _read(agent_path)
        assert "from .rollout_status import classify_rollout_status" in source
        assert "rollout_status = classify_rollout_status(" in source
        for field in (
            "overlong",
            "no_finish",
            "hit_token_limit",
            "hit_max_turn",
            "hit_timeout",
            "unfolded_main",
            "termination_reason",
        ):
            assert f"'{field}': rollout_status['{field}']" in source


def test_sab_method_comparison_uses_a_fixed_sample_seed():
    source = _read("scripts/eval_sab_react_8b_4node_smoke.sh")
    assert "SAB_DATA_SEED=${SAB_DATA_SEED:-42}" in source
    assert "data.seed=$SAB_DATA_SEED" in source


def test_sab_formal_wrapper_submits_all_three_methods_with_real_eval():
    source = _read("scripts/submit_eval_sab_8b_4node_formal.sh")
    assert 'if [ -n "${SLURM_JOB_ID:-}" ]; then' in source
    assert "for method in react fold ctxgraph; do" in source
    assert "SAB_RUN_TAG=formal" in source
    assert "SAB_REAL_EVAL=1" in source
    assert "SAB_VAL_MAX_SAMPLES=${SAB_VAL_MAX_SAMPLES:--1}" in source
    assert '--nodes=4' in source
    assert 'JOB_TIME=${JOB_TIME:-04:00:00}' in source
