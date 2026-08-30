from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_openthinker_submit_wires_model_to_bc_and_gaia():
    source = _read("scripts/submit_eval_openthinker_sft_8b_bc_gaia.sh")

    assert "open-thoughts/OpenThinkerAgent-8B-ColdStartSFTForRL" in source
    assert "scripts/submit_eval_bc_8b_4node_zeroshot_64k.sh" in source
    assert "scripts/submit_gaia_benchmark_8b_5node.sh" in source
    assert "scripts/check_hf_model_support.py" in source
    assert 'PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in source
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in source
    assert "BC_EXPERIMENT_MODEL_TAG=openthinker_sft_8b" in source
    assert "GAIA_EXPERIMENT_MODEL_TAG=openthinker_sft_8b" in source


def test_base_submitters_propagate_model_identity():
    bc_submit = _read("scripts/submit_eval_bc_8b_4node_zeroshot_64k.sh")
    bc_eval = _read("scripts/eval_bc_baseline_8b_4node_zeroshot.sh")
    gaia_submit = _read("scripts/submit_gaia_benchmark_8b_5node.sh")

    assert 'MODEL_PATH="$MODEL_PATH"' in bc_submit
    assert 'BC_EXPERIMENT_MODEL_TAG="$BC_EXPERIMENT_MODEL_TAG"' in bc_submit
    assert "BC_EXPERIMENT_MODEL_TAG=${BC_EXPERIMENT_MODEL_TAG:-8b}" in bc_eval
    assert "MODEL_PATH=$GAIA_MODEL_PATH" in gaia_submit
    assert "GAIA_EXPERIMENT_MODEL_TAG" in gaia_submit


def test_openthinker_idev_runner_is_single_sample_controller_smoke():
    source = _read("scripts/eval_openthinker_sft_8b_idev.sh")

    assert "OpenThinkerAgent-8B-ColdStartSFTForRL" in source
    assert 'if [ "${#NODELIST[@]}" -ne 4 ]; then' in source
    assert "export CONDA_ENV_NAME=deepseek_v4 SEARCH_CONDA_ENV_NAME=cxtgraph" in source
    assert "export EXPECTED_NUM_NODES=4 BC_METHOD=contextgraph BC_CTXGRAPH_PROTOCOL=controller" in source
    assert "EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-1}" in source
    assert "export BC_VAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES BC_ROLLOUT_N=1" in source
    assert "export BC_CONTEXT_LENGTH=32768 BC_PROMPT_LENGTH=8192 BC_RESPONSE_LENGTH=24576" in source
    assert "export BC_ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1" in source
    assert "unset LORA_ADAPTER_PATH LORA_RANK LORA_ALPHA" in source
    assert "exec bash scripts/eval_bc_baseline_8b_4node_zeroshot.sh" in source
