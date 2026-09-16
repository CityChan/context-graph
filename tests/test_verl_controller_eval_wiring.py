from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_vllm_server_adapts_wire_schema_to_v010_guided_decoding():
    source = _read("verl/workers/rollout/vllm_rollout/vllm_async_server.py")
    assert "from vllm.sampling_params import GuidedDecodingParams" in source
    assert 'sampling_params.pop("structured_outputs", None)' in source
    assert "build_vllm_guided_decoding(" in source
    assert 'sampling_params["guided_decoding"]' in source


def test_call_llm_forwards_structured_outputs_without_mutating_base_sampling():
    source = _read("agents/utils.py")
    assert "structured_outputs = kwargs.pop('structured_outputs', None)" in source
    assert "sampling_params = dict(self.sampling_params)" in source
    assert "normalize_structured_outputs(structured_outputs)" in source
    assert "structured_outputs and guided_decoding cannot both be set" in source


def test_sab_code_graph_agent_executes_controller_owned_checkpoints():
    source = _read("agents/graph_agent_code_isolated.py")
    assert "GraphActionController" in source
    assert "expose_graph_tools=not structured_graph_controller" in source
    assert '"structured_outputs": graph_controller.structured_outputs(' in source
    assert "and not structured_graph_controller" in source
    assert "graph_controller.action_prompt(" in source
    assert "graph_controller.resolve_action(" in source
    assert "[GRAPH CONTROLLER {controller_action.upper()}]" in source
    assert "env.stats['structured_graph_controller']" in source


def test_controller_mode_rejections_do_not_pollute_graph_invalid_ops():
    for path in (
        "agents/graph_agent_isolated.py",
        "agents/graph_agent_code_isolated.py",
    ):
        source = _read(path)
        rejection_block = source.split(
            "controller_mode_rejections += 1", 1
        )[1].split(
            "elif fn_call is not None and fn_call['function'] in GRAPH_OPS", 1
        )[0]
        assert "graph.record_graph_op(False)" not in rejection_block
        assert "[GRAPH CONTROLLER MODE REJECTION]" in rejection_block
        assert "env.stats['controller_mode_rejections']" in source
        assert "env.stats['controller_mode_rejection_rate']" in source
        assert "controller_action_policy" in source
        assert "env.stats['controller_structural_policy']" in source

    reward_manager = _read("verl/workers/reward_manager/agent.py")
    assert '"controller_mode_rejections"' in reward_manager
    assert '"controller_mode_rejection_rate"' in reward_manager
    assert '"controller_structural_policy"' in reward_manager


def test_alfworld_controller_supports_non_destructive_long_horizon_ablation():
    source = _read("agents/graph_agent_isolated.py")
    assert 'getattr(config.plugin, "controller_allow_pass", False)' in source
    assert "controller_allow_pass or graph.is_saturated()" in source
    assert 'getattr(config.plugin, "inject_graph_state_after_action", True)' in source
    assert "if inject_graph_state_after_action:" in source
    assert "enable_history_replacement = enable_retrieval_memory and not is_train" in source
    assert "env.stats['controller_allow_pass']" in source
    assert "env.stats['memory_graph_state_after_action']" in source


def test_sab_8b_eval_has_opt_in_controller_protocol():
    source = _read("scripts/eval_sab_react_8b_4node_smoke.sh")
    assert "SAB_CTXGRAPH_PROTOCOL=${SAB_CTXGRAPH_PROTOCOL:-legacy}" in source
    assert "SAB_CTXGRAPH_PROTOCOL=controller requires SAB_METHOD=ctxgraph" in source
    assert "plugin.structured_graph_controller=$SAB_STRUCTURED_GRAPH_CONTROLLER" in source
    assert "plugin.controller_owned_tool_formatting=$SAB_CONTROLLER_OWNED_TOOL_FORMATTING" in source
    assert "plugin.controller_action_policy=$SAB_CONTROLLER_ACTION_POLICY" in source
    assert "GuidedDecodingParams" in source


def test_browsecomp_8b_eval_has_opt_in_controller_protocol():
    source = _read("scripts/eval_bc_baseline_8b_4node_zeroshot.sh")
    assert "BC_CTXGRAPH_PROTOCOL=${BC_CTXGRAPH_PROTOCOL:-legacy}" in source
    assert "BC_CTXGRAPH_PROTOCOL=controller requires BC_METHOD=contextgraph" in source
    assert "plugin.structured_graph_controller=$BC_STRUCTURED_GRAPH_CONTROLLER" in source
    assert "plugin.controller_owned_tool_formatting=$BC_CONTROLLER_OWNED_TOOL_FORMATTING" in source
    assert "plugin.controller_action_policy=$BC_CONTROLLER_ACTION_POLICY" in source
    assert "GuidedDecodingParams" in source


def test_sab_30b_eval_and_submitter_preserve_protocol_identity():
    runner = _read("scripts/eval_sab_react_30b_instruct_8node_smoke.sh")
    submitter = _read("scripts/submit_sab_react_30b_instruct_8node_formal.sh")
    assert "SAB_CTXGRAPH_PROTOCOL=${SAB_CTXGRAPH_PROTOCOL:-legacy}" in runner
    assert "plugin.structured_graph_controller=$SAB_STRUCTURED_GRAPH_CONTROLLER" in runner
    assert "plugin.controller_owned_tool_formatting=$SAB_CONTROLLER_OWNED_TOOL_FORMATTING" in runner
    assert "plugin.controller_action_policy=$SAB_CONTROLLER_ACTION_POLICY" in runner
    assert 'SAB_CTXGRAPH_PROTOCOL="$SAB_CTXGRAPH_PROTOCOL"' in submitter
    assert "${SAB_METHOD}_${SAB_CTXGRAPH_PROTOCOL}_sab" in submitter


def test_matched_suite_submitters_apply_controller_protocol_only_to_ctxgraph():
    sab = _read("scripts/submit_eval_sab_8b_4node_formal.sh")
    bc = _read("scripts/submit_eval_bc_8b_4node_zeroshot_64k.sh")
    for source in (sab, bc):
        assert "method_protocol=legacy" in source
        assert "method_protocol=" in source
    assert 'SAB_CTXGRAPH_PROTOCOL="$method_protocol"' in sab
    assert 'BC_CTXGRAPH_PROTOCOL="$method_protocol"' in bc
    assert "BC_CTXGRAPH_PROTOCOL=${BC_CTXGRAPH_PROTOCOL:-controller}" in bc


def test_discovery_submitter_enables_structural_controller_policy():
    submitter = _read("scripts/submit_eval_discoverybench_qwen3_8b_4node.sh")
    assert (
        "DISCOVERYBENCH_CONTROLLER_ACTION_POLICY="
        "${DISCOVERYBENCH_CONTROLLER_ACTION_POLICY:-structural}"
    ) in submitter
    assert (
        "SAB_CONTROLLER_ACTION_POLICY="
        "$DISCOVERYBENCH_CONTROLLER_ACTION_POLICY"
    ) in submitter


def test_gaia_submitter_and_runners_support_controller_protocol_and_sample_caps():
    submitter = _read("scripts/submit_gaia_benchmark_8b_5node.sh")
    assert "GAIA_CTXGRAPH_PROTOCOL=${GAIA_CTXGRAPH_PROTOCOL:-controller}" in submitter
    assert "method_protocol=legacy" in submitter
    assert 'if [ "$method" = "ctxgraph" ]' in submitter
    assert "BC_CTXGRAPH_PROTOCOL=$method_protocol" in submitter
    assert "BC_CONTROLLER_ACTION_POLICY=$GAIA_CONTROLLER_ACTION_POLICY" in submitter
    assert "TRAIN_MAX_SAMPLES=$GAIA_TRAIN_MAX_SAMPLES" in submitter
    assert "VAL_MAX_SAMPLES=$GAIA_VAL_MAX_SAMPLES" in submitter
    assert 'if [ "$DRY_RUN" = "1" ]' in submitter

    runners = (
        "scripts/train_bc_baseline_8b_4node_24h_v3_32k.sh",
        "scripts/train_bc_foldagent_8b_paperfaithful_5node_48h.sh",
        "scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh",
    )
    for path in runners:
        source = _read(path)
        assert 'data.train_max_samples="$TRAIN_MAX_SAMPLES"' in source
        assert 'data.val_max_samples="$VAL_MAX_SAMPLES"' in source

    ctxgraph = _read("scripts/train_bc_ctxgraph_8b_paperfaithful_5node_48h.sh")
    assert "BC_CTXGRAPH_PROTOCOL=${BC_CTXGRAPH_PROTOCOL:-legacy}" in ctxgraph
    assert "plugin.structured_graph_controller=\"$BC_STRUCTURED_GRAPH_CONTROLLER\"" in ctxgraph
    assert "plugin.controller_owned_tool_formatting=\"$BC_CONTROLLER_OWNED_TOOL_FORMATTING\"" in ctxgraph
    assert "plugin.controller_action_policy=\"$BC_CONTROLLER_ACTION_POLICY\"" in ctxgraph


def test_browsecomp_30b_contextgraph_evals_have_controller_protocol():
    for path in (
        "scripts/eval_bc_ctxgraph_30b_instruct_8node_zeroshot.sh",
        "scripts/eval_bc_ctxgraph_30b_8node_zeroshot.sh",
    ):
        source = _read(path)
        assert "CTXGRAPH_PROTOCOL=${CTXGRAPH_PROTOCOL:-legacy}" in source
        assert "plugin.structured_graph_controller=$STRUCTURED_GRAPH_CONTROLLER" in source
        assert "plugin.controller_owned_tool_formatting=$CONTROLLER_OWNED_TOOL_FORMATTING" in source
        assert "GuidedDecodingParams" in source
