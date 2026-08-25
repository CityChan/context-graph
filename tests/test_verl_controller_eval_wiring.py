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


def test_sab_8b_eval_has_opt_in_controller_protocol():
    source = _read("scripts/eval_sab_react_8b_4node_smoke.sh")
    assert "SAB_CTXGRAPH_PROTOCOL=${SAB_CTXGRAPH_PROTOCOL:-legacy}" in source
    assert "SAB_CTXGRAPH_PROTOCOL=controller requires SAB_METHOD=ctxgraph" in source
    assert "plugin.structured_graph_controller=$SAB_STRUCTURED_GRAPH_CONTROLLER" in source
    assert "plugin.controller_owned_tool_formatting=$SAB_CONTROLLER_OWNED_TOOL_FORMATTING" in source
    assert "GuidedDecodingParams" in source


def test_browsecomp_8b_eval_has_opt_in_controller_protocol():
    source = _read("scripts/eval_bc_baseline_8b_4node_zeroshot.sh")
    assert "BC_CTXGRAPH_PROTOCOL=${BC_CTXGRAPH_PROTOCOL:-legacy}" in source
    assert "BC_CTXGRAPH_PROTOCOL=controller requires BC_METHOD=contextgraph" in source
    assert "plugin.structured_graph_controller=$BC_STRUCTURED_GRAPH_CONTROLLER" in source
    assert "plugin.controller_owned_tool_formatting=$BC_CONTROLLER_OWNED_TOOL_FORMATTING" in source
    assert "GuidedDecodingParams" in source


def test_sab_30b_eval_and_submitter_preserve_protocol_identity():
    runner = _read("scripts/eval_sab_react_30b_instruct_8node_smoke.sh")
    submitter = _read("scripts/submit_sab_react_30b_instruct_8node_formal.sh")
    assert "SAB_CTXGRAPH_PROTOCOL=${SAB_CTXGRAPH_PROTOCOL:-legacy}" in runner
    assert "plugin.structured_graph_controller=$SAB_STRUCTURED_GRAPH_CONTROLLER" in runner
    assert "plugin.controller_owned_tool_formatting=$SAB_CONTROLLER_OWNED_TOOL_FORMATTING" in runner
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
